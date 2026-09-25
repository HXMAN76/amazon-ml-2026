"""xenc: a cross-encoder for the uncertain band, used as one extra feature of the stack.

The pair model compares two records through hand-made similarities; its remaining errors are name noise (injected or dropped words,
typos, aliases) and look-alike distractors. A cross-encoder reads both records together and can weigh such differences directly. It is
trained only on the uncertain band of the first-stage probability p1 (plus confident false positives as hard negatives and a sample of
confident positives), so inference touches only about 2% of the candidates.

  python -m ber.stages.xenc data  [--base v5]   -> WORK/xenc/{train_fit,train,test}.parquet   (ber env)
  python -m ber.stages.xenc train                -> WORK/xenc/model                             (pytorch env, GPU)
  python -m ber.stages.xenc score --split S      -> WORK/xenc/{S}_xs.parquet (q, pid, xs)       (pytorch env, GPU)

The S1 used to fit the cross-encoder are excluded from stage-two training (their xs would look better than on new data), the same
guard as for the dense encoders. The base encoder is intfloat/multilingual-e5-small (MIT, 118M parameters) with a one-logit head.
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q
from ber.stages.block import PID_BASE

MODEL = "intfloat/multilingual-e5-small"
DATA_DIR = "xenc"  # WORK sub-folder of the pair lists and scores (`--dir xenc_v7` keeps a second set); the fitted model always lives in xenc/model
DIGITS = re.compile(r"\d+")


def tag_digits(t: str) -> str:
    """Wrap digit runs in markers so that a changed or dropped digit is one visible difference, not an arbitrary subword split."""
    return DIGITS.sub(lambda m: f"[{'P' if len(m.group()) >= 5 else 'N'}]{m.group()}[/]", t)


def xenc_train_q(prm: dict) -> np.ndarray:
    """S1 for fitting the cross-encoder: outside the stage-1 sample, the holdout and the S1 the dense encoders were fitted on."""
    from ber.stages.dense import dense_train_q
    from ber.stages.dense_all import dense_all_train_q

    P = config.paths()
    cfg = config.load()
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
    ok = np.setdiff1d(s1, np.concatenate([smp, holdout_q().astype(np.int64), dense_train_q(cfg["dense"]), dense_all_train_q(cfg["dense_all"])]))
    rng = np.random.default_rng(prm["seed"])
    return np.sort(rng.choice(ok, size=min(prm["fit_s1"], len(ok)), replace=False))


def _texts(split: str):
    """Text of every S1 (by rid) and pool record (S2 then S3): name and address, with the romanised name for non-Latin names."""
    pq = config.paths()["parquet"] / split

    def col(f):
        d = pl.read_parquet(pq / f, columns=["rid", "name1", "addr", "core_rom", "nl_name"]).sort("rid")
        t = pl.when(pl.col("nl_name") > 0.5).then(pl.col("name1") + " (" + pl.col("core_rom") + ")").otherwise(pl.col("name1"))
        return (t + " | " + pl.col("addr")).alias("t"), d

    out = []
    for f in ("source1.parquet", "source2.parquet", "source3.parquet"):
        expr, d = col(f)
        out.append(d.select(expr)["t"].to_numpy())
    return out[0], np.concatenate([out[1], out[2]]), len(out[1])


def _pairs(split: str, base: str, prm: dict) -> pl.DataFrame:
    from ber.stages.stack import load_p1, shortlist

    d = shortlist(load_p1(split, base), config.load()["stack"])
    return d.select("q", "pid", "p", *([c for c in ("label",) if c in d.columns]))


def data(base: str) -> None:
    P = config.paths()
    prm = config.load()["xenc"]
    out = P["work"] / DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = prm["band_lo"], prm["band_hi"]
    for split in ("train", "test"):
        t0 = time.time()
        d = _pairs(split, base, prm)
        band = d.filter((pl.col("p") >= lo) & (pl.col("p") <= hi))
        s1t, poolt, n2 = _texts(split)

        def with_text(x: pl.DataFrame) -> pl.DataFrame:
            pid = x["pid"].to_numpy()
            pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
            return x.with_columns(pl.Series("ta", [tag_digits(t) for t in s1t[x["q"].to_numpy()]]),
                                  pl.Series("tb", [tag_digits(t) for t in poolt[pidx]]))

        with_text(band).write_parquet(out / f"{split}.parquet", compression="zstd")
        print(f"{split}: {band.height} band pairs to score ({band.height / d['q'].n_unique():.2f} per S1) in {time.time() - t0:.0f}s", flush=True)
        if split == "train":
            fit_q = pl.DataFrame({"q": xenc_train_q(prm)})
            rng = np.random.default_rng(prm["seed"])
            f = d.join(fit_q, on="q", how="semi")
            neg = f.filter((pl.col("label") == 0) & (pl.col("p") > 0.9))          # confident false positives: the look-alikes
            pos = f.filter((pl.col("label") == 1) & (pl.col("p") > hi))
            pos = pos.filter(pl.Series(rng.random(pos.height) < prm["easy_pos_frac"]))
            fit = pl.concat([f.filter((pl.col("p") >= lo) & (pl.col("p") <= hi)), neg, pos]).unique(["q", "pid"])
            with_text(fit).write_parquet(out / "train_fit.parquet", compression="zstd")
            print(f"fit set: {fit.height} pairs ({int(fit['label'].sum())} positive), band {f.filter((pl.col('p') >= lo) & (pl.col('p') <= hi)).height}, "
                  f"confident false positives {neg.height}, easy positives {pos.height}", flush=True)


def train() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    P = config.paths()
    prm = config.load()["xenc"]
    t0 = time.time()
    d = pl.read_parquet(P["work"] / DATA_DIR / "train_fit.parquet")
    ta, tb, y = d["ta"].to_list(), d["tb"].to_list(), d["label"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=prm["lr"], weight_decay=0.01)
    bs, epochs = prm["batch"], prm["epochs"]
    steps = epochs * (len(y) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.06 * steps)) * max(0.0, 1 - i / steps))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    model.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(y))
        for a in range(0, len(order) - bs + 1, bs):
            idx = order[a: a + bs]
            enc = tok([ta[i] for i in idx], [tb[i] for i in idx], padding=True, truncation=True, max_length=prm["max_len"], return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[idx]).cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = P["work"] / "xenc" / "model"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} after {time.time() - t0:.0f}s", flush=True)


def score(split: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    P = config.paths()
    prm = config.load()["xenc"]
    t0 = time.time()
    d = pl.read_parquet(P["work"] / DATA_DIR / f"{split}.parquet")
    tok = AutoTokenizer.from_pretrained(P["work"] / "xenc" / "model")
    model = AutoModelForSequenceClassification.from_pretrained(P["work"] / "xenc" / "model").cuda().half().eval()
    ta, tb = d["ta"].to_list(), d["tb"].to_list()
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
    xs = np.zeros(len(order), dtype=np.float32)
    bs = prm["score_batch"]
    with torch.no_grad():
        for a in range(0, len(order), bs):
            idx = order[a: a + bs]
            enc = tok([ta[i] for i in idx], [tb[i] for i in idx], padding=True, truncation=True, max_length=prm["max_len"], return_tensors="pt").to("cuda")
            xs[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    out = d.select("q", "pid").with_columns(pl.Series("xs", xs))
    if "label" in d.columns:
        from sklearn.metrics import average_precision_score
        print(f"{split}: average precision of xs {average_precision_score(d['label'].to_numpy(), xs):.4f} versus p1 {average_precision_score(d['label'].to_numpy(), d['p'].to_numpy()):.4f} inside the band", flush=True)
    out.write_parquet(P["work"] / DATA_DIR / f"{split}_xs.parquet", compression="zstd")
    print(f"{split}: {out.height} pairs scored in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "score"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--base", default="v5")
    ap.add_argument("--dir", default="xenc", help="WORK sub-folder for pair lists and scores")
    a = ap.parse_args(argv)
    global DATA_DIR
    DATA_DIR = a.dir
    {"data": lambda: data(a.base), "train": train, "score": lambda: score(a.split)}[a.cmd]()


if __name__ == "__main__":
    main()
