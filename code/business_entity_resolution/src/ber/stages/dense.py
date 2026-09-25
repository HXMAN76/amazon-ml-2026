"""dense: multilingual name-embedding retrieval for pool records written in non-Latin scripts.

Token blocking finds a record through shared words. A pool record whose name is in Devanagari, Telugu, Malayalam, ... shares
no word with its Latin S1 name and often has only a short, generic Latin address, so blocking misses many of these pairs (about
12% of S1 have such a match, and their oracle F_0.5 is 0.92 against 0.99 for the rest). This stage embeds S1 names and
non-Latin pool names with a multilingual sentence encoder (intfloat/multilingual-e5-small, MIT, 118M parameters, weights
downloaded once; no data is looked up) and adds, for every non-Latin pool record, its nearest S1 names as extra candidates.

  python -m ber.stages.dense finetune                     -> WORK/dense/model_ft (contrastive fine-tuning on true pairs)
  python -m ber.stages.dense embed --split train|test     -> WORK/dense/{split}/{s1,pool}_emb.npy (+ ids)
  python -m ber.stages.dense retrieve --split train|test  -> WORK/dense/{split}/pairs.parquet (q, pid, cos, rank)
  python -m ber.stages.dense report                       -> recall of true non-Latin pairs: current candidates vs dense union
  python -m ber.stages.dense merge --split train|test     -> adds missing dense pairs to WORK/blocks/{split} shards

`embed` needs torch and transformers (the g5 `pytorch` conda env); everything else runs in the normal env.
"""

from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.block import PID_BASE
from ber.tracking import log_stage

MODEL = "intfloat/multilingual-e5-small"


def encode(texts: list[str], batch: int = 512, max_len: int = 32, device: str = "cuda") -> np.ndarray:
    """L2-normalised sentence embeddings (float16) with the "query: " prefix on both sides (symmetric matching)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = config.paths()["work"] / "dense" / "model_ft"
    src = str(path) if (path / "config.json").exists() else MODEL  # the fine-tuned encoder, when there is one
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src).to(device).half().eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = order[s: s + batch]
            enc = tok(["query: " + texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out[idx] = e.half().cpu().numpy()
    return out


def topk_pairs(s_emb: np.ndarray, s_ids: np.ndarray, p_emb: np.ndarray, p_ids: np.ndarray, k: int, chunk: int = 1024,
               device: str | None = None) -> pl.DataFrame:
    """For every pool embedding the k most similar S1 embeddings (cosine): rows (q, pid, cos, rank)."""
    try:
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        S = torch.from_numpy(s_emb).to(dev)
        S = S.half() if dev == "cuda" else S.float()
        qs, cs = [], []
        for a in range(0, len(p_emb), chunk):
            P = torch.from_numpy(p_emb[a: a + chunk]).to(dev)
            P = P.half() if dev == "cuda" else P.float()
            v, i = (P @ S.T).topk(min(k, S.shape[0]), dim=1)
            qs.append(i.cpu().numpy())
            cs.append(v.float().cpu().numpy())
        idx, cos = np.concatenate(qs), np.concatenate(cs)
    except ImportError:  # numpy fallback (tests)
        S = s_emb.astype(np.float32)
        idx_l, cos_l = [], []
        for a in range(0, len(p_emb), chunk):
            sim = p_emb[a: a + chunk].astype(np.float32) @ S.T
            i = np.argsort(-sim, axis=1)[:, :k]
            idx_l.append(i)
            cos_l.append(np.take_along_axis(sim, i, axis=1))
        idx, cos = np.concatenate(idx_l), np.concatenate(cos_l)
    kk = idx.shape[1]
    return pl.DataFrame({"q": s_ids[idx.ravel()].astype(np.int64), "pid": np.repeat(p_ids, kk).astype(np.int64),
                         "cos": cos.ravel().astype(np.float32), "rank": np.tile(np.arange(kk, dtype=np.int16), len(p_ids))})


def _texts(split: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """India S1 (rid, name) and non-Latin India pool records (pid, name) of a split."""
    P = config.paths()
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "ctry"]).filter(pl.col("ctry") == "india")
    pools = []
    for src in (2, 3):
        d = pl.read_parquet(pq / f"source{src}.parquet", columns=["rid", "name1", "ctry", "nl_name"]).filter(
            (pl.col("ctry") == "india") & (pl.col("nl_name") > 0.5))
        pools.append(d.with_columns((pl.col("rid").cast(pl.Int64) + src * PID_BASE).alias("pid")))
    pool = pl.concat(pools)
    return s1["rid"].to_numpy().astype(np.int64), s1["name1"].to_list(), pool["pid"].to_numpy(), pool["name1"].to_list()


def dense_train_q(prm: dict) -> np.ndarray:
    """India S1 used to fine-tune the encoder: drawn from those outside BOTH the stage-1 sample and the locked holdout."""
    from ber.stages.score_rest import holdout_q

    P = config.paths()
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).filter(pl.col("ctry") == "india")
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])
    ok = np.setdiff1d(s1["rid"].to_numpy().astype(np.int64), np.concatenate([smp["rid"].to_numpy().astype(np.int64), holdout_q().astype(np.int64)]))
    rng = np.random.default_rng(prm["seed"])
    return np.sort(rng.choice(ok, size=min(prm["ft_s1"], len(ok)), replace=False))


def _pool_texts_by_pid(pq, ids: np.ndarray) -> dict[int, str]:
    out: dict[int, str] = {}
    for src in (2, 3):
        d = pl.read_parquet(pq / f"source{src}.parquet", columns=["rid", "name1"])
        for rid, nm in zip(d["rid"].to_list(), d["name1"].to_list()):
            out[src * PID_BASE + rid] = nm
    return {int(i): out[int(i)] for i in ids}


def finetune() -> None:
    """Supervised contrastive fine-tuning of the encoder on true (S1 name, non-Latin pool name) pairs of the dense-train S1.

    Symmetric InfoNCE with in-batch negatives, one random true pair per S1 and epoch (so a batch never holds two positives of
    the same S1). Weights are saved under WORK/dense/model_ft and picked up by `encode`.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    P = config.paths()
    prm = config.load()["dense"]
    t0 = time.time()
    pq = P["parquet"] / "train"
    D = dense_train_q(prm)
    np.save(P["work"] / "dense" / "train_q.npy", D)
    _, _, p_ids, _ = _texts("train")
    lab = pl.read_parquet(pq / "labels.parquet").with_columns((pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
                                                              pl.col("s1_rid").alias("q")).select("q", "pid")
    pairs = lab.join(pl.DataFrame({"q": D}), on="q", how="semi").join(pl.DataFrame({"pid": p_ids}), on="pid", how="semi")
    s1n = dict(zip(*[pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1"])[c].to_list() for c in ("rid", "name1")]))
    ptxt = _pool_texts_by_pid(pq, pairs["pid"].to_numpy())
    groups = pairs.group_by("q").agg(pl.col("pid")).to_dict(as_series=False)
    qs, plist = groups["q"], groups["pid"]
    print(f"fine-tuning on {len(qs)} S1 with {pairs.height} true non-Latin pairs", flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=prm["ft_lr"], weight_decay=0.01)
    bs, epochs = prm["ft_batch"], prm["ft_epochs"]
    steps = epochs * (len(qs) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.05 * steps)) * max(0.0, 1 - i / steps))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)

    def emb(texts):
        enc = tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)

    model.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(qs))
        for a in range(0, len(order) - bs + 1, bs):
            b = order[a: a + bs]
            ta = [s1n[int(qs[i])] for i in b]
            tb = [ptxt[int(plist[i][rng.integers(len(plist[i]))])] for i in b]
            ea, eb = emb(ta), emb(tb)
            logits = ea @ eb.T / prm["ft_temp"]
            lab_t = torch.arange(len(b), device=dev)
            loss = (torch.nn.functional.cross_entropy(logits, lab_t) + torch.nn.functional.cross_entropy(logits.T, lab_t)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = P["work"] / "dense" / "model_ft"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} after {time.time() - t0:.0f}s", flush=True)


def embed(split: str) -> None:
    """Encode S1 and pool names and cache them under WORK/dense/{split}."""
    P = config.paths()
    t0 = time.time()
    s_ids, s_txt, p_ids, p_txt = _texts(split)
    out = P["work"] / "dense" / split
    out.mkdir(parents=True, exist_ok=True)
    print(f"{split}: {len(s_txt)} India S1 names, {len(p_txt)} non-Latin pool names", flush=True)
    np.save(out / "s1_ids.npy", s_ids)
    np.save(out / "pool_ids.npy", p_ids)
    np.save(out / "s1_emb.npy", encode(s_txt))
    print(f"S1 encoded in {time.time() - t0:.0f}s", flush=True)
    np.save(out / "pool_emb.npy", encode(p_txt))
    print(f"pool encoded, total {time.time() - t0:.0f}s", flush=True)


def retrieve(split: str, k: int | None = None) -> None:
    """Nearest S1 names for every non-Latin pool record; writes pairs.parquet."""
    P = config.paths()
    prm = config.load()["dense"]
    d = P["work"] / "dense" / split
    pairs = topk_pairs(np.load(d / "s1_emb.npy"), np.load(d / "s1_ids.npy"), np.load(d / "pool_emb.npy"), np.load(d / "pool_ids.npy"), k or prm["k"])
    pairs.write_parquet(d / "pairs.parquet", compression="zstd")
    print(f"{split}: {pairs.height} dense pairs, mean cosine of rank 1: {pairs.filter(pl.col('rank') == 0)['cos'].mean():.3f}", flush=True)


def report(prm: dict | None = None) -> dict:
    """Recall of the true non-Latin (S1, pool) pairs: current candidates versus the dense union (train)."""
    P = config.paths()
    prm = prm or config.load()["dense"]
    pairs = pl.read_parquet(P["work"] / "dense" / "train" / "pairs.parquet")
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q")).select("q", "pid")
    _, _, p_ids, _ = _texts("train")
    truth = lab.join(pl.DataFrame({"pid": p_ids}), on="pid", how="semi")  # true pairs whose pool record is non-Latin (India)
    from ber.stages.score_rest import holdout_q

    truth = truth.join(pl.DataFrame({"q": holdout_q()}), on="q", how="semi")  # only S1 the encoder never saw (locked holdout)
    cand = pl.concat([pl.read_parquet(f, columns=["q", "pid"]) for f in sorted((P["work"] / "blocks" / "train").glob("cand_*.parquet"))])
    cand = cand.join(pl.DataFrame({"q": holdout_q()}), on="q", how="semi")
    inc = truth.join(cand.with_columns(pl.lit(1).alias("c")), on=["q", "pid"], how="left")["c"].is_not_null()
    rep = {"true_nonlatin_pairs": truth.height, "recall_current_candidates": float(inc.mean())}
    for k in (1, 3, 5, 10, 20):
        hit = truth.join(pairs.filter(pl.col("rank") < k).with_columns(pl.lit(1).alias("d")), on=["q", "pid"], how="left")["d"].is_not_null()
        rep[f"recall_dense_top{k}"] = float(hit.mean())
        rep[f"recall_union_top{k}"] = float((inc | hit).mean())
        rep[f"new_pairs_per_pool_record_top{k}"] = float(k)
    print("DENSE REPORT:", rep, flush=True)
    return rep


def merge(split: str) -> None:
    """Append the dense pairs that the blocker did not propose to the candidate shards (score columns zero, emb_cos set)."""
    P = config.paths()
    prm = config.load()["dense"]
    pairs = pl.read_parquet(P["work"] / "dense" / split / "pairs.parquet").filter(pl.col("rank") < prm["k_merge"])
    src = P["work"] / "blocks" / split
    bak = P["work"] / "blocks" / f"{split}_prededense"
    if bak.exists():  # keep the untouched shards; always merge from them
        shutil.rmtree(src)
        shutil.copytree(bak, src)
    else:
        shutil.copytree(src, bak)
    total_new = 0
    for f in sorted(src.glob("cand_*.parquet")):
        c = pl.read_parquet(f)
        qmin, qmax = int(c["q"].min()), int(c["q"].max())
        # the shard's S1 range is [qmin, qmax]; dense pairs of S1 inside it join this shard
        d = (pairs.filter((pl.col("q") >= qmin) & (pl.col("q") <= qmax))
                  .select(pl.col("q").cast(c.schema["q"]), pl.col("pid").cast(c.schema["pid"]), pl.col("cos").alias("emb_cos"),
                          pl.col("rank").cast(pl.Float32).alias("emb_rank")))
        m = c.join(d, on=["q", "pid"], how="left")
        new = d.join(c.select("q", "pid"), on=["q", "pid"], how="anti")
        if new.height:
            zeros = {col: pl.lit(0.0 if c.schema[col].is_float() else 0).cast(c.schema[col]) for col in c.columns if col not in ("q", "pid")}
            new = new.with_columns(**zeros).select(m.columns)
            m = pl.concat([m, new])
        total_new += new.height
        m.sort("q", "pid").write_parquet(f, compression="zstd")
    print(f"{split}: {total_new} dense pairs added to the candidate shards", flush=True)
    log_stage(f"dense_merge_{split}", prm, {"added": float(total_new)})


def main(argv: list[str] | None = None) -> None:
    """CLI: embed | retrieve | report | merge (see module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "embed", "retrieve", "report", "merge"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    a = ap.parse_args(argv)
    {"finetune": finetune, "embed": lambda: embed(a.split), "retrieve": lambda: retrieve(a.split), "report": report, "merge": lambda: merge(a.split)}[a.cmd]()


if __name__ == "__main__":
    main()
