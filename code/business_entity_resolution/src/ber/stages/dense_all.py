"""dense_all: name + address embeddings for every S2/S3 record, used as a second candidate channel next to token blocking.

Token blocking misses pairs whose names are glued (`blackstonselect`), scrambled by typos (`rajkot usnioinr`), reduced to a generic word
or written in another script. This stage fine-tunes the multilingual encoder on true (S1, pool) pairs of all countries, using one mined
hard negative per pair (a non-matching candidate of the same S1), embeds "name | address" of every S1 and pool record, and retrieves for
every pool record its nearest S1 records (a pool record has at most one owner, so this direction fits the task).

  python -m ber.stages.dense_all finetune
  python -m ber.stages.dense_all embed --split train|test       -> WORK/dense_all/{split}/{s1,pool}_emb.npy (+ ids)
  python -m ber.stages.dense_all retrieve --split train|test    -> WORK/dense_all/{split}/pairs.parquet (q, pid, cos, rank)
  python -m ber.stages.dense_all report                         -> holdout recall gain versus number of pairs added, per (k, tau)
  python -m ber.stages.dense_all merge --split train|test       -> adds pairs to the candidate shards, feature columns dall_cos, dall_rank

Needs torch and transformers (the `pytorch` conda env) except for `report` and `merge`.
"""

from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.block import PID_BASE
from ber.stages.dense import MODEL, topk_pairs
from ber.tracking import log_stage

TEXT = (pl.col("name1") + " | " + pl.col("addr")).alias("text")


def model_dir():
    return config.paths()["work"] / "dense_all" / "model_ft"


def _load_texts(split: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """(S1 rid, text) and (pool pid, text) for all records of a split; text is the name and the normalised address."""
    pq = config.paths()["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "addr"]).with_columns(TEXT)
    pools = [pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "addr"]).with_columns(
        TEXT, (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]
    pool = pl.concat(pools)
    return s1["rid"].to_numpy().astype(np.int64), s1["text"].to_list(), pool["pid"].to_numpy().astype(np.int64), pool["text"].to_list()


def dense_all_train_q(prm: dict) -> np.ndarray:
    """S1 used to fine-tune: outside the stage-1 sample and the locked holdout, with at least one true match."""
    from ber.split import holdout_q

    P = config.paths()
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet", columns=["s1_rid"]).unique()
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])
    ok = np.setdiff1d(lab["s1_rid"].to_numpy().astype(np.int64), np.concatenate([smp["rid"].to_numpy().astype(np.int64), holdout_q().astype(np.int64)]))
    rng = np.random.default_rng(prm["seed"])
    return np.sort(rng.choice(ok, size=min(prm["ft_s1"], len(ok)), replace=False))


def _embedder(src: str, max_len: int, device: str = "cuda", train: bool = False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src).to(device)

    def emb(texts):
        enc = tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)

    return tok, model, emb


def finetune() -> None:
    """Symmetric InfoNCE over (S1, true pool record) with one hard negative per S1 (a non-true candidate of the same S1)."""
    import torch

    P = config.paths()
    prm = config.load()["dense_all"]
    t0 = time.time()
    D = dense_all_train_q(prm)
    np.save(P["work"] / "dense_all_train_q.npy", D)
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q")).select("q", "pid")
    lab = lab.join(pl.DataFrame({"q": D}), on="q", how="semi")
    dq = pl.DataFrame({"q": D})
    cand = pl.concat([pl.read_parquet(f, columns=["q", "pid"]).join(dq, on="q", how="semi")
                      for f in sorted((P["work"] / "blocks" / "train").glob("cand_*.parquet"))])
    neg = cand.join(lab, on=["q", "pid"], how="anti").group_by("q").agg(pl.col("pid"))
    pos = lab.group_by("q").agg(pl.col("pid"))
    both = pos.join(neg, on="q", how="inner", suffix="_neg")
    qs, ppos, pneg = both["q"].to_list(), both["pid"].to_list(), both["pid_neg"].to_list()
    sid, stx, pid, ptx = _load_texts("train")
    s_of = dict(zip(sid.tolist(), stx))
    need = {int(x) for l in ppos for x in l} | {int(x) for l in pneg for x in l}
    p_of = {int(i): t for i, t in zip(pid.tolist(), ptx) if int(i) in need}
    del stx, ptx
    print(f"fine-tuning on {len(qs)} S1 (pairs with hard negatives)", flush=True)

    tok, model, emb = _embedder(MODEL, prm["max_len"], train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=prm["ft_lr"], weight_decay=0.01)
    bs, epochs = prm["ft_batch"], prm["ft_epochs"]
    steps = epochs * (len(qs) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.05 * steps)) * max(0.0, 1 - i / steps))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    model.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(qs))
        for a in range(0, len(order) - bs + 1, bs):
            b = order[a: a + bs]
            ta = [s_of[qs[i]] for i in b]
            tp = [p_of[int(ppos[i][rng.integers(len(ppos[i]))])] for i in b]
            tn = [p_of[int(pneg[i][rng.integers(len(pneg[i]))])] for i in b]
            ea, ep_, en = emb(ta), emb(tp), emb(tn)
            pool = torch.cat([ep_, en])                      # in-batch positives plus one hard negative per S1
            logits = ea @ pool.T / prm["ft_temp"]
            tgt = torch.arange(len(b), device=ea.device)
            loss = (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy((ea @ ep_.T).T / prm["ft_temp"], tgt)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = model_dir()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} after {time.time() - t0:.0f}s", flush=True)


def _encode_to(path, texts: list[str], emb, batch: int = 1024, block: int = 262_144, dim: int = 384) -> None:
    """Encode texts into a float16 .npy memmap block by block (length-sorted inside each block)."""
    import torch

    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), dim))
    with torch.no_grad():
        for a in range(0, len(texts), block):
            seg = texts[a: a + block]
            order = np.argsort([len(t) for t in seg])
            buf = np.zeros((len(seg), dim), dtype=np.float16)
            for s in range(0, len(seg), batch):
                idx = order[s: s + batch]
                buf[idx] = emb([seg[i] for i in idx]).half().cpu().numpy()
            out[a: a + len(seg)] = buf
    out.flush()


def embed(split: str) -> None:
    P = config.paths()
    prm = config.load()["dense_all"]
    t0 = time.time()
    _, _, emb = _embedder(str(model_dir()), prm["max_len"])
    sid, stx, pid, ptx = _load_texts(split)
    out = P["work"] / "dense_all" / split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "s1_ids.npy", sid)
    np.save(out / "pool_ids.npy", pid)
    print(f"{split}: {len(stx)} S1 and {len(ptx)} pool records to encode", flush=True)
    _encode_to(out / "s1_emb.npy", stx, emb)
    print(f"S1 encoded in {time.time() - t0:.0f}s", flush=True)
    _encode_to(out / "pool_emb.npy", ptx, emb)
    print(f"pool encoded, total {time.time() - t0:.0f}s", flush=True)


def retrieve(split: str) -> None:
    P = config.paths()
    prm = config.load()["dense_all"]
    t0 = time.time()
    d = P["work"] / "dense_all" / split
    pairs = topk_pairs(np.load(d / "s1_emb.npy"), np.load(d / "s1_ids.npy"), np.load(d / "pool_emb.npy", mmap_mode="r"),
                       np.load(d / "pool_ids.npy"), prm["k"], chunk=1024)
    pairs.write_parquet(d / "pairs.parquet", compression="zstd")
    print(f"{split}: {pairs.height} pairs, mean cosine of rank 1 {pairs.filter(pl.col('rank') == 0)['cos'].mean():.3f} "
          f"in {time.time() - t0:.0f}s", flush=True)


def _existing(split: str) -> pl.DataFrame:
    P = config.paths()
    return pl.concat([pl.read_parquet(f, columns=["q", "pid"]) for f in sorted((P["work"] / "blocks" / split).glob("cand_*.parquet"))])


def report() -> None:
    """Holdout: how many currently missed true pairs each (k, tau) recovers, and how many pairs it adds per S1."""
    from ber.split import holdout_q

    P = config.paths()
    pairs = pl.read_parquet(P["work"] / "dense_all" / "train" / "pairs.parquet")
    hq = pl.DataFrame({"q": holdout_q()})
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q")).select("q", "pid")
    truth = lab.join(hq, on="q", how="semi")
    cand = _existing("train")
    cand_h = cand.join(hq, on="q", how="semi")
    missed = truth.join(cand_h, on=["q", "pid"], how="anti")
    n_s1 = cand["q"].n_unique()
    print(f"holdout true pairs {truth.height}, missed by current candidates {missed.height} ({missed.height / truth.height:.4f})", flush=True)
    for k in (1, 3, 5):
        for tau in (0.0, 0.8, 0.85, 0.9):
            sel = pairs.filter((pl.col("rank") < k) & (pl.col("cos") >= tau))
            new = sel.join(cand, on=["q", "pid"], how="anti")
            found = missed.join(sel, on=["q", "pid"], how="semi").height
            print(f"k={k} tau={tau:.2f}: adds {new.height} pairs ({new.height / n_s1:.2f} per S1), recovers {found} of {missed.height} missed "
                  f"({found / truth.height:.4f} of all true pairs)", flush=True)


def merge(split: str) -> None:
    """Add dense-all pairs (rank < k_merge, cos >= tau) to the candidate shards; every pair gets dall_cos and dall_rank."""
    P = config.paths()
    prm = config.load()["dense_all"]
    pairs = pl.read_parquet(P["work"] / "dense_all" / split / "pairs.parquet").filter((pl.col("rank") < prm["k_merge"]) & (pl.col("cos") >= prm["tau"]))
    src = P["work"] / "blocks" / split
    bak = P["work"] / "blocks" / f"{split}_predall"
    if bak.exists():
        shutil.rmtree(src)
        shutil.copytree(bak, src)
    else:
        shutil.copytree(src, bak)
    total_new = 0
    for f in sorted(src.glob("cand_*.parquet")):
        c = pl.read_parquet(f)
        qmin, qmax = int(c["q"].min()), int(c["q"].max())
        d = (pairs.filter((pl.col("q") >= qmin) & (pl.col("q") <= qmax))
                  .select(pl.col("q").cast(c.schema["q"]), pl.col("pid").cast(c.schema["pid"]), pl.col("cos").alias("dall_cos"),
                          pl.col("rank").cast(pl.Float32).alias("dall_rank")))
        m = c.join(d, on=["q", "pid"], how="left")
        new = d.join(c.select("q", "pid"), on=["q", "pid"], how="anti")
        if new.height:
            fill = {col: pl.lit(0.0 if c.schema[col].is_float() else 0).cast(c.schema[col]) for col in c.columns if col not in ("q", "pid", "emb_cos", "emb_rank")}
            fill.update({col: pl.lit(None, dtype=c.schema[col]) for col in ("emb_cos", "emb_rank") if col in c.columns})
            new = new.with_columns(**fill).select(m.columns)
            m = pl.concat([m, new])
        total_new += new.height
        m.sort("q", "pid").write_parquet(f, compression="zstd")
    print(f"{split}: {total_new} dense-all pairs added to the candidate shards", flush=True)
    log_stage(f"dense_all_merge_{split}", prm, {"added": float(total_new)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "embed", "retrieve", "report", "merge"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    a = ap.parse_args(argv)
    {"finetune": finetune, "embed": lambda: embed(a.split), "retrieve": lambda: retrieve(a.split), "report": report,
     "merge": lambda: merge(a.split)}[a.cmd]()


if __name__ == "__main__":
    main()
