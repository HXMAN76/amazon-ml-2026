"""xenc: fine-tuned cross-encoder on the uncertain band of the first-stage probabilities; its score `xs` is a stacking feature.

The pair features compare strings one measure at a time. A cross-encoder reads both records together, which helps where the features
disagree: cross-script names, a changed digit in an otherwise identical address, abbreviations the rules do not know. It runs only on
pairs the first stage is unsure about (p1 in [lo, hi]), so its cost is bounded. Training data follows Ditto (Li et al., VLDB 2021):
  * hard negatives: besides the band, every confident false positive (label 0, p1 > hi) of the training S1, i.e. the look-alike
    distractors, plus a sample of confident positives so both classes stay represented;
  * domain tagging: digit runs are wrapped as [POSTCODE]..[/POSTCODE] (5+ digits) or [NUM]..[/NUM], so a changed house number is
    a visible event instead of an arbitrary sub-word split; non-Latin names also carry their romanised copy.
  * sibling context (v6): the S1 side also carries the S1's most confident OTHER candidate (other source first, p1 >= 0.5), so a
    look-alike (`house 323` against the confirmed sibling's `312`) is visible in one input: `S1 [SIB] sibling` versus the candidate;
  * synthetic look-alikes (v6, ber.decoys): a copy of confident true records with a few letters and the house number moved, label 0.
The S1 used for fine-tuning are saved to WORK/xenc/train_q.npy and are kept out of stage-two training (their xs would be optimistic);
they are drawn first from the S1 the dense encoders were fine-tuned on, which stacking excludes already.

  python -m ber.stages.xenc finetune --base v5a                  -> WORK/xenc/model, WORK/xenc/train_q.npy
  python -m ber.stages.xenc score --split train|test --base v5a  -> WORK/xenc/{split}/scores.parquet (q, pid, xs), band pairs only
  python -m ber.stages.xenc report                                -> holdout: AUC-PR of xs versus p1 inside the band
`finetune` and `score` need torch, transformers, sentence-transformers and datasets (requirements-gpu.txt).
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.block import PID_BASE
from ber.tracking import log_stage

_DIGITS = re.compile(r"\d+")


def tag_digits(s: str) -> str:
    """`12 Oak St 06103` -> `[NUM]12[/NUM] Oak St [POSTCODE]06103[/POSTCODE]`."""
    return _DIGITS.sub(lambda m: f"[POSTCODE]{m.group()}[/POSTCODE]" if len(m.group()) >= 5 else f"[NUM]{m.group()}[/NUM]", s or "")


def record_text(name: str, addr: str, rom: str, nonlatin: float) -> str:
    """Cross-encoder input for one record: raw name (+ romanised copy for Indic scripts) | digit-tagged raw address."""
    n = (name or "").strip()
    if nonlatin > 0 and rom:
        n = f"{n} ({rom})"
    return f"{n} | {tag_digits((addr or '').strip())}"


def siblings(p1: pl.DataFrame, min_p: float = 0.5) -> pl.DataFrame:
    """(q, pid, sib): for every pair, the S1's most confident other candidate with p1 >= min_p, from the other source if there is one."""
    d = p1.select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "p").with_columns((pl.col("pid") // PID_BASE).alias("src"))
    conf = d.filter(pl.col("p") >= min_p)
    per_src = conf.sort("p", descending=True).group_by("q", "src").agg(pl.col("pid").head(2).alias("top"))
    other = per_src.with_columns((5 - pl.col("src")).alias("src"))  # the other source's best records, keyed by this pair's source
    same = per_src
    out = d.join(other.rename({"top": "o"}), on=["q", "src"], how="left").join(same.rename({"top": "s"}), on=["q", "src"], how="left")
    s0, s1 = pl.col("s").list.get(0, null_on_oob=True), pl.col("s").list.get(1, null_on_oob=True)
    same_sib = pl.when(s0 == pl.col("pid")).then(s1).otherwise(s0)  # the best same-source record that is not the pair itself
    return out.select("q", "pid", pl.coalesce(pl.col("o").list.get(0, null_on_oob=True), same_sib).alias("sib"))


def with_sibling(s_of: dict[int, str], p_of: dict[int, str], q: int, sib) -> str:
    """S1 text, plus the sibling's text after [SIB] when the S1 has a confident other candidate."""
    return s_of[q] if sib is None else f"{s_of[q]} [SIB] {p_of[int(sib)]}"


def band(p1: pl.DataFrame, lo: float, hi: float) -> pl.DataFrame:
    """Pairs the first stage is unsure about."""
    return p1.filter((pl.col("p") >= lo) & (pl.col("p") <= hi))


def training_rows(p1: pl.DataFrame, train_q: np.ndarray, lo: float, hi: float, n_pos_extra: int, max_pairs: int, seed: int) -> pl.DataFrame:
    """Band pairs of the fine-tuning S1 + all their confident false positives (hard negatives) + sampled confident positives."""
    d = p1.join(pl.DataFrame({"q": train_q}).with_columns(pl.col("q").cast(p1.schema["q"])), on="q", how="semi")
    b = band(d, lo, hi)
    hard_neg = d.filter((pl.col("p") > hi) & (pl.col("label") == 0))
    pos = d.filter((pl.col("p") > hi) & (pl.col("label") == 1))
    pos = pos.sample(n=min(n_pos_extra, pos.height), seed=seed) if pos.height else pos
    rows = pl.concat([b, hard_neg, pos]).unique(subset=["q", "pid"])
    if rows.height > max_pairs:
        rows = rows.sample(n=max_pairs, seed=seed)
    return rows.sort("q", "pid")


def xenc_train_q(all_q: np.ndarray, hold: np.ndarray, preferred: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Fine-tuning S1: outside the locked holdout, taken first from `preferred` (S1 stacking excludes anyway), then at random."""
    rng = np.random.default_rng(seed)
    pref = np.setdiff1d(np.intersect1d(preferred, all_q), hold)
    take = rng.choice(pref, size=min(n, len(pref)), replace=False) if len(pref) else np.array([], dtype=np.int64)
    if len(take) < n:
        rest = np.setdiff1d(np.setdiff1d(all_q, hold), take)
        take = np.concatenate([take, rng.choice(rest, size=min(n - len(take), len(rest)), replace=False)])
    return np.sort(take.astype(np.int64))


def _texts(split: str, pairs: pl.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    """Cross-encoder text of the S1 (by rid) and pool records (by pid) that appear in `pairs` (q, pid): bounded memory."""
    pq = config.paths()["parquet"] / split
    cols = ["rid", "business_name", "business_address", "core_rom", "nl_name"]
    to_text = lambda d: [record_text(*r) for r in d.select("business_name", "business_address", "core_rom", "nl_name").iter_rows()]  # noqa: E731
    need_q = pairs.select(pl.col("q").cast(pl.Int64).alias("rid")).unique()
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).with_columns(pl.col("rid").cast(pl.Int64)).join(need_q, on="rid", how="semi")
    s_of = dict(zip(s1["rid"].to_list(), to_text(s1)))
    p_of: dict[int, str] = {}
    need_p = pairs.select(pl.col("pid").cast(pl.Int64)).unique()
    for s in (2, 3):
        d = (pl.read_parquet(pq / f"source{s}.parquet", columns=cols).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"))
               .join(need_p, on="pid", how="semi"))
        p_of.update(zip(d["pid"].to_list(), to_text(d)))
    return s_of, p_of


def _preferred_q() -> np.ndarray:
    """S1 the dense encoders were fine-tuned on (stage two excludes them already), when those stages are configured."""
    cfg = config.load()
    out = [np.array([], dtype=np.int64)]
    f2 = config.paths()["work"] / "dense_all2_train_q.npy"
    if f2.exists():  # v6: the second channel's S1 first (the largest set stage two leaves out)
        out.append(np.load(f2).astype(np.int64))
    if cfg.get("dense"):
        from ber.stages.dense import dense_train_q

        out.append(dense_train_q(cfg["dense"]).astype(np.int64))
    if cfg.get("dense_all"):
        from ber.stages.dense_all import dense_all_train_q

        out.append(dense_all_train_q(cfg["dense_all"]).astype(np.int64))
    return np.unique(np.concatenate(out))


def finetune(base: str) -> None:
    """Fine-tune the cross-encoder (binary cross-entropy) on the band + hard negatives of the fine-tuning S1."""
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from ber.split import holdout_q
    from ber.stages.stack import load_p1

    P = config.paths()
    prm = config.load()["xenc"]
    t0 = time.time()
    p1 = load_p1("train", base)
    all_q = p1["q"].unique().to_numpy().astype(np.int64)
    from ber.split import unscored_q

    tq = xenc_train_q(all_q, np.union1d(holdout_q().astype(np.int64), unscored_q()), _preferred_q(), prm["train_s1"], prm["seed"])
    out = P["work"] / "xenc"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_q.npy", tq)
    rows = training_rows(p1, tq, prm["lo"], prm["hi"], prm["n_pos_extra"], prm["max_pairs"], prm["seed"])
    if prm.get("sibling", False):
        rows = rows.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(
            siblings(p1.join(pl.DataFrame({"q": tq}).with_columns(pl.col("q").cast(p1.schema["q"])), on="q", how="semi")), on=["q", "pid"], how="left")
    else:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Int64).alias("sib"))
    s_of, p_of = _texts("train", pl.concat([rows.select("q", "pid"), rows.filter(pl.col("sib").is_not_null()).select("q", pl.col("sib").alias("pid"))]))
    a = [with_sibling(s_of, p_of, int(q), sb) for q, sb in zip(rows["q"].to_list(), rows["sib"].to_list())]
    b = [p_of[int(p)] for p in rows["pid"].to_list()]
    y = rows["label"].cast(pl.Float32).to_list()
    n_dec = int(prm.get("n_synth_decoy", 0))
    if n_dec:  # look-alikes of true records as extra negatives, same S1 side (with its sibling)
        from ber.decoys import make_decoy

        rng = np.random.default_rng(prm["seed"])
        posi = [i for i, v in enumerate(y) if v == 1.0]
        for i in rng.choice(posi, size=min(n_dec, len(posi)), replace=False):
            name, addr = (b[i].split(" | ", 1) + [""])[:2]
            a.append(a[i])
            b.append(" | ".join(make_decoy(name, addr, rng)))
            y.append(0.0)
    print(f"fine-tuning on {rows.height} pairs of {len(tq)} S1 (positives {int(sum(y))})", flush=True)
    model = CrossEncoder(prm["base_model"], num_labels=1, max_length=prm["max_len"], trust_remote_code=True)
    ds = Dataset.from_dict({"text1": a, "text2": b, "label": y})
    args = CrossEncoderTrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=prm["epochs"], per_device_train_batch_size=prm["batch"],
                                         learning_rate=prm["lr"], warmup_ratio=0.1, fp16=True, dataloader_num_workers=0, save_strategy="no",
                                         logging_steps=200, report_to="none", seed=prm["seed"])
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=BinaryCrossEntropyLoss(model))
    trainer.train()
    model.save_pretrained(str(out / "model"))
    if not any((out / "model").iterdir()):
        raise SystemExit("cross-encoder weights were not written")
    print(f"saved {out / 'model'} after {time.time() - t0:.0f}s", flush=True)
    log_stage("xenc_finetune", prm, {"pairs": float(rows.height), "s1": float(len(tq)), "seconds": time.time() - t0})


def score(split: str, base: str) -> None:
    """Score every band pair of a split (train: except the fine-tuning S1) with the fine-tuned cross-encoder."""
    from sentence_transformers.cross_encoder import CrossEncoder

    from ber.stages.stack import load_p1

    P = config.paths()
    prm = config.load()["xenc"]
    t0 = time.time()
    p1 = load_p1(split, base)
    d = band(p1, prm["lo"], prm["hi"])
    if split == "train":
        d = d.join(pl.DataFrame({"q": np.load(P["work"] / "xenc" / "train_q.npy")}).with_columns(pl.col("q").cast(d.schema["q"])),
                   on="q", how="anti")
    d = d.select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    if prm.get("sibling", False):
        d = d.join(siblings(p1.join(d.select("q").unique().with_columns(pl.col("q").cast(p1.schema["q"])), on="q", how="semi")), on=["q", "pid"], how="left")
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Int64).alias("sib"))
    s_of, p_of = _texts(split, pl.concat([d.select("q", "pid"), d.filter(pl.col("sib").is_not_null()).select("q", pl.col("sib").alias("pid"))]))
    model = CrossEncoder(str(P["work"] / "xenc" / "model"), max_length=prm["max_len"], trust_remote_code=True)
    pairs = [(with_sibling(s_of, p_of, int(q), sb), p_of[int(p)]) for q, p, sb in zip(d["q"].to_list(), d["pid"].to_list(), d["sib"].to_list())]
    order = np.argsort([len(x) + len(y) for x, y in pairs])  # length-sorted batches waste less padding
    xs = np.empty(len(pairs), dtype=np.float32)
    step = 200_000
    for s in range(0, len(order), step):
        idx = order[s: s + step]
        xs[idx] = np.asarray(model.predict([pairs[i] for i in idx], batch_size=prm["score_batch"], show_progress_bar=False), dtype=np.float32)
        print(f"{split}: scored {min(s + step, len(order))}/{len(order)} band pairs ({time.time() - t0:.0f}s)", flush=True)
    out = P["work"] / "xenc" / split
    out.mkdir(parents=True, exist_ok=True)
    d.select("q", "pid").with_columns(pl.Series("xs", xs)).write_parquet(out / "scores.parquet", compression="zstd")
    log_stage(f"xenc_score_{split}", prm, {"pairs": float(len(pairs)), "seconds": time.time() - t0})


def report(base: str) -> dict:
    """Locked holdout, inside the band: does xs rank true pairs better than p1 (average precision)?"""
    from sklearn.metrics import average_precision_score

    from ber.split import holdout_q
    from ber.stages.stack import load_p1

    P = config.paths()
    prm = config.load()["xenc"]
    xs = pl.read_parquet(P["work"] / "xenc" / "train" / "scores.parquet")
    d = band(load_p1("train", base), prm["lo"], prm["hi"]).join(
        pl.DataFrame({"q": holdout_q()}).with_columns(pl.col("q").cast(pl.Int64)), on="q", how="semi")
    d = d.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(xs, on=["q", "pid"], how="inner")
    y = d["label"].to_numpy()
    rep = {"band_pairs": d.height, "positives": int(y.sum()), "ap_p1": float(average_precision_score(y, d["p"].to_numpy())),
           "ap_xs": float(average_precision_score(y, d["xs"].to_numpy()))}
    prev = P["work"] / "xenc_v1" / "train" / "scores.parquet"
    if prev.exists():  # the previous cross-encoder on the holdout pairs both scored
        c = d.join(pl.read_parquet(prev).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xs_prev")),
                   on=["q", "pid"], how="inner")
        yc = c["label"].to_numpy()
        rep.update(common_pairs=c.height, ap_p1_common=float(average_precision_score(yc, c["p"].to_numpy())),
                   ap_xs_common=float(average_precision_score(yc, c["xs"].to_numpy())),
                   ap_xs_prev_common=float(average_precision_score(yc, c["xs_prev"].to_numpy())))
    print("XENC REPORT (locked holdout, band only):", rep, flush=True)
    log_stage("xenc_report", prm, rep)
    return rep


def main(argv: list[str] | None = None) -> None:
    """CLI: finetune | score | report (see module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "score", "report"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--base", default=None, help="first-stage model supplying p1 (default: xenc.base)")
    a = ap.parse_args(argv)
    base = a.base or config.load()["xenc"]["base"]
    {"finetune": lambda: finetune(base), "score": lambda: score(a.split, base), "report": lambda: report(base)}[a.cmd]()


if __name__ == "__main__":
    main()
