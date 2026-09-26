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
import json
import re
import time

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q
from ber.stages.block import PID_BASE

MODEL = "intfloat/multilingual-e5-small"
DATA_DIR = "xenc"  # WORK sub-folder of the pair lists and scores (`--dir xenc_v7` keeps a second set); the fitted model always lives in xenc/model
MODEL_DIR = "xenc/model"  # WORK-relative folder of the fitted cross-encoder (set by --model-dir)
OVERRIDES: dict = {}   # parameter overrides from --set
DIGITS = re.compile(r"\d+")


def _prm() -> dict:
    prm = dict(config.load()["xenc"])
    for k, v in OVERRIDES.items():
        prm[k] = type(prm[k])(v) if k in prm else float(v)
    return prm


def is_decoder(name: str) -> bool:
    """Decoder-only bases (Qwen3) have no separator token and classify from the last token, so the pair is formatted and closed explicitly."""
    return "qwen" in str(name).lower()


def encode_pairs(tok, ta: list[str], tb: list[str], max_len: int, decoder: bool):
    """Model inputs for a batch of record pairs (cuda tensors). Encoders use the tokenizer's pair encoding; decoders get one prompt per
    pair, truncated to max_len - 1 tokens and closed by the end token so that the last-token classification head always reads the same token."""
    import torch

    if not decoder:
        return tok(ta, tb, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
    ids = tok([f"Same business?\nA: {a}\nB: {b}" for a, b in zip(ta, tb)], add_special_tokens=False, truncation=True, max_length=max_len - 1)["input_ids"]
    end, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id
    n = max(len(x) for x in ids) + 1
    inp = torch.full((len(ids), n), pad, dtype=torch.long)
    att = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x) + 1] = torch.tensor(x + [end])
        att[i, :len(x) + 1] = 1
    return {"input_ids": inp.cuda(), "attention_mask": att.cuda()}


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
    first = rng.choice(ok, size=min(prm["fit_s1"], len(ok)), replace=False)
    if prm.get("fit_more"):  # a larger fit set that contains the first one (so earlier scores stay valid)
        rest = np.setdiff1d(ok, first)
        first = np.concatenate([first, np.random.default_rng(prm["seed"] + 1).choice(rest, size=min(int(prm["fit_more"]), len(rest)), replace=False)])
    return np.sort(first)


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
    prm = _prm()
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
    prm = _prm()
    t0 = time.time()
    d = pl.read_parquet(P["work"] / DATA_DIR / "train_fit.parquet")
    if OVERRIDES.get("max_rows"):  # smoke test: a small random subset
        d = d.sample(int(OVERRIDES["max_rows"]), seed=0)
    ta, tb, y = d["ta"].to_list(), d["tb"].to_list(), d["label"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    dec = is_decoder(MODEL)
    sym = int(prm.get("symmetric", 0))
    if dec:
        tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, **({"dtype": torch.float32} if dec else {}))
    model.config.pad_token_id = tok.pad_token_id
    if dec:  # 0.6B parameters in fp32 with Adam leave too little room for activations on a 24 GB GPU otherwise
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = model.cuda()
    amp = torch.bfloat16 if dec else torch.float16
    opt = torch.optim.AdamW(model.parameters(), lr=prm["lr"], weight_decay=0.01)
    bs, epochs = prm["batch"], prm["epochs"]
    steps = epochs * (len(y) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.06 * steps)) * max(0.0, 1 - i / steps))
    scaler = torch.amp.GradScaler(enabled=not dec)
    rng = np.random.default_rng(0)
    model.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(y))
        for a in range(0, len(order) - bs + 1, bs):
            idx = order[a: a + bs]
            xa, xb = [ta[i] for i in idx], [tb[i] for i in idx]
            if sym:  # symmetric training: the pair is shown in either order, so the score cannot depend on which record comes first
                flip = rng.random(len(idx)) < 0.5
                xa, xb = [b if f else a for a, b, f in zip(xa, xb, flip)], [a if f else b for a, b, f in zip(xa, xb, flip)]
            enc = encode_pairs(tok, xa, xb, prm["max_len"], dec)
            with torch.autocast("cuda", dtype=amp):
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
    out = P["work"] / MODEL_DIR
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} after {time.time() - t0:.0f}s", flush=True)


def score(split: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    P = config.paths()
    prm = _prm()
    t0 = time.time()
    d = pl.read_parquet(P["work"] / DATA_DIR / f"{split}.parquet")
    if OVERRIDES.get("max_rows"):  # smoke test
        d = d.head(int(OVERRIDES["max_rows"]))
    tok = AutoTokenizer.from_pretrained(P["work"] / MODEL_DIR)
    dec = is_decoder(json.loads((P["work"] / MODEL_DIR / "config.json").read_text()).get("model_type", ""))
    model = AutoModelForSequenceClassification.from_pretrained(P["work"] / MODEL_DIR).cuda().to(torch.bfloat16 if dec else torch.float16).eval()
    ta, tb = d["ta"].to_list(), d["tb"].to_list()
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
    xs = np.zeros(len(order), dtype=np.float32)
    asym = np.zeros(len(order), dtype=np.float32)
    sym = int(prm.get("symmetric", 0))
    bs = prm["score_batch"]
    with torch.no_grad():
        for a in range(0, len(order), bs):
            idx = order[a: a + bs]
            xa, xb = [ta[i] for i in idx], [tb[i] for i in idx]
            l1 = model(**encode_pairs(tok, xa, xb, prm["max_len"], dec)).logits.squeeze(-1).float()
            if sym:  # average the logits of both orders; the difference is a measure of how unsure the model is
                l2 = model(**encode_pairs(tok, xb, xa, prm["max_len"], dec)).logits.squeeze(-1).float()
                asym[idx] = (l1 - l2).abs().cpu().numpy()
                l1 = (l1 + l2) / 2
            xs[idx] = torch.sigmoid(l1).cpu().numpy()
    out = d.select("q", "pid").with_columns(pl.Series("xs", xs))
    if sym:
        out = out.with_columns(pl.Series("xs_asym", asym))
    if "label" in d.columns:
        from sklearn.metrics import average_precision_score
        print(f"{split}: average precision of xs {average_precision_score(d['label'].to_numpy(), xs):.4f} versus p1 {average_precision_score(d['label'].to_numpy(), d['p'].to_numpy()):.4f} inside the band", flush=True)
    out.write_parquet(P["work"] / DATA_DIR / f"{split}_xs.parquet", compression="zstd")
    print(f"{split}: {out.height} pairs scored in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    global DATA_DIR, MODEL, MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "score"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--base", default="v5")
    ap.add_argument("--dir", default="xenc", help="WORK sub-folder for pair lists and scores")
    ap.add_argument("--base-model", default=MODEL, help="Hugging Face encoder to fine-tune (train)")
    ap.add_argument("--model-dir", default="xenc/model", help="WORK-relative folder of the fitted model")
    ap.add_argument("--set", default="", help="comma-separated xenc parameter overrides, e.g. epochs=3,band_lo=0.01")
    a = ap.parse_args(argv)
    DATA_DIR, MODEL, MODEL_DIR = a.dir, a.base_model, a.model_dir
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        OVERRIDES[k] = v
    {"data": lambda: data(a.base), "train": train, "score": lambda: score(a.split)}[a.cmd]()


if __name__ == "__main__":
    main()
