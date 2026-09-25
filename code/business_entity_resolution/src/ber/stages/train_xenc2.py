"""train_xenc2: cross-encoder v2 (v4.md stage 4) — hard-negative mining + numeric/postcode tagging.

Same base model and CrossEncoderTrainer setup as train_xenc.py (stage 2b), validated working this
session. Two additions from Ditto (ARCHITECTURE_v3_research.md §2), not present in stage 2b:

1. Hard-negative mining: stage 2b trained on a random sample of the uncertain band. Here we also
   mine near-miss negatives directly from the feature table — pairs with label=0 but pin_match>0
   or house_eq>0 (same postcode or house number, different entity) — the exact confusions the
   uncertain band's random sample under-represents.
2. Numeric/postcode tagging: digit runs in the input text are wrapped in [NUM]...[/NUM] markers
   before tokenization, so digit transposition/typos get an explicit signal instead of being split
   arbitrarily by the subword tokenizer.

Inputs : WORK/models/<baseline>/oof.parquet (uncertain-band selection, same as stage 2b)
         WORK/features/train/part_*.parquet (label, pin_match, house_eq columns for hard-neg mining)
         WORK/parquet/train/source{1,2,3}.parquet (raw text)
Outputs: WORK/models/<name>/xenc/ (fine-tuned cross-encoder weights)
         WORK/models/<name>/xenc_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import polars as pl

from ber import config
from ber.stages.train_xenc import BASE_MODEL, BASE_MODEL_LICENSE, BASE_MODEL_PARAMS_APPROX, _pool_text
from ber.tracking import log_stage

_NUM_RE = re.compile(r"\d+")


def _tag_numbers(text: str) -> str:
    """Wrap digit runs in [NUM]...[/NUM] so house numbers/postcodes get an explicit signal."""
    return _NUM_RE.sub(lambda m: f"[NUM]{m.group()}[/NUM]", text)


def main(argv: list[str] | None = None) -> None:
    """CLI: fine-tune the cross-encoder on uncertain-band + mined hard negatives, with digit tagging."""
    prm = config.load().get("xenc", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xenc2")
    ap.add_argument("--baseline", default="v0base")
    ap.add_argument("--lo", type=float, default=prm.get("lo", 0.15))
    ap.add_argument("--hi", type=float, default=prm.get("hi", 0.85))
    ap.add_argument("--base-model", default=prm.get("base_model", BASE_MODEL))
    ap.add_argument("--epochs", type=int, default=prm.get("epochs", 1))
    ap.add_argument("--batch-size", type=int, default=prm.get("batch_size", 64))
    ap.add_argument("--max-pairs", type=int, default=prm.get("max_pairs", 200_000))
    ap.add_argument("--hard-neg-frac", type=float, default=0.2, help="fraction of the training set that is mined hard negatives")
    ap.add_argument("--seed", type=int, default=prm.get("seed", 0))
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()

    oof = pl.read_parquet(P["work"] / "models" / a.baseline / "oof.parquet")
    band = oof.filter((pl.col("p") >= a.lo) & (pl.col("p") <= a.hi))
    n_band = min(band.height, int(a.max_pairs * (1 - a.hard_neg_frac)))
    if band.height > n_band:
        band = band.sample(n_band, seed=a.seed)

    n_hard = int(a.max_pairs * a.hard_neg_frac)
    feat = pl.read_parquet(str(P["work"] / "features" / "train" / "part_*.parquet"),
                            columns=["q", "pid", "label", "pin_match", "house_eq"])
    hard = feat.filter((pl.col("label") == 0) & ((pl.col("pin_match") > 0) | (pl.col("house_eq") > 0)))
    hard = hard.select("q", "pid", "label")
    if hard.height > n_hard:
        hard = hard.sample(n_hard, seed=a.seed)
    hard = hard.join(band.select("q", "pid"), on=["q", "pid"], how="anti")  # no duplicates with the band sample

    train = pl.concat([band.select("q", "pid", "label"), hard])
    print(f"training set: {band.height} uncertain-band pairs (positives {int(band['label'].sum())}) "
          f"+ {hard.height} mined hard negatives = {train.height} total", flush=True)
    if train.height == 0:
        raise SystemExit("empty training set: widen --lo/--hi or check --baseline's OOF distribution")

    pq = P["parquet"] / "train"
    cols = ["rid", "core1", "addr"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=cols)
    s3 = pl.read_parquet(pq / "source3.parquet", columns=cols)
    pool = pl.concat([s2, s3])

    q = train["q"].to_numpy()
    pid = train["pid"].to_numpy()
    a_text = [_tag_numbers(t) for t in (s1[q]["core1"] + " " + s1[q]["addr"]).to_list()]
    b_text = [_tag_numbers(t) for t in _pool_text(pool, pid, s2.height)]
    labels = train["label"].to_numpy().astype(np.float32)

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(a.base_model, num_labels=1, trust_remote_code=True, max_length=128)
    train_dataset = Dataset.from_dict({"sentence1": a_text, "sentence2": b_text, "label": labels.tolist()})
    loss = BinaryCrossEntropyLoss(model)
    out = P["work"] / "models" / a.name / "xenc"
    out.mkdir(parents=True, exist_ok=True)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=a.seed,
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
    trainer.train()
    model.save_pretrained(str(out))

    cfg = {"base_model": a.base_model, "base_model_license": BASE_MODEL_LICENSE,
           "base_model_params_approx": BASE_MODEL_PARAMS_APPROX, "baseline": a.baseline,
           "lo": a.lo, "hi": a.hi, "n_pairs": train.height, "n_hard_negatives": hard.height,
           "epochs": a.epochs, "seed": a.seed, "numeric_tagging": True}
    (P["work"] / "models" / a.name).mkdir(parents=True, exist_ok=True)
    (P["work"] / "models" / a.name / "xenc_config.json").write_text(json.dumps(cfg, indent=2))
    log_stage("train_xenc2", cfg, {"pairs": float(train.height), "seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s, saved to {out}", flush=True)


if __name__ == "__main__":
    main()
