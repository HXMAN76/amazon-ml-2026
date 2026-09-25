"""train_xenc: fine-tune a cross-encoder on the uncertain band of train pairs (v2.md stage 2b).

Base model: Alibaba-NLP/gte-multilingual-reranker-base (Apache-2.0, ~305M params, offline, encoder-only).
License and size logged to xenc_config.json for the audit trail; fine-tuning uses only this competition's
train pairs, no external labelled data.

Only pairs whose baseline pair-model probability p1 falls in [lo, hi] are used: these are exactly the
cases a pointwise feature classifier is unsure about, where token-level cross-attention (typos, digit
transpositions, word-order swaps, non-Latin/romanised name variants) can add signal beyond bag-of-features
string similarity. Training on the confident 0/1 tails would waste the fine-tune budget on pairs the
existing matcher already gets right.

Inputs : WORK/models/<baseline>/oof.parquet (q, pid, p, label) for the dev sample, OOF so no leakage
         WORK/parquet/train/source{1,2,3}.parquet (raw text: core1, addr)
Outputs: WORK/models/<name>/xenc/ (fine-tuned cross-encoder weights)
         WORK/models/<name>/xenc_config.json (base model, license, band, pair count)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config
from ber.tracking import log_stage

BASE_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
BASE_MODEL_LICENSE = "Apache-2.0"
BASE_MODEL_PARAMS_APPROX = "305M"


def _pool_text(pool: pl.DataFrame, pid: np.ndarray, n2: int) -> list[str]:
    from ber.stages.pairs import pool_index

    idx = pool_index(pool, pid, n2)
    rows = pool[idx]
    return (rows["core1"] + " " + rows["addr"]).to_list()


def main(argv: list[str] | None = None) -> None:
    """CLI: fine-tune the cross-encoder on the uncertain band of a baseline model's OOF pairs."""
    prm = config.load().get("xenc", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xenc0")
    ap.add_argument("--baseline", default="v0base", help="pair model whose OOF p selects the uncertain band")
    ap.add_argument("--lo", type=float, default=prm.get("lo", 0.15))
    ap.add_argument("--hi", type=float, default=prm.get("hi", 0.85))
    ap.add_argument("--base-model", default=prm.get("base_model", BASE_MODEL))
    ap.add_argument("--epochs", type=int, default=prm.get("epochs", 1))
    ap.add_argument("--batch-size", type=int, default=prm.get("batch_size", 64))
    ap.add_argument("--max-pairs", type=int, default=prm.get("max_pairs", 200_000), help="cap for the fine-tune time budget")
    ap.add_argument("--seed", type=int, default=prm.get("seed", 0))
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()

    oof = pl.read_parquet(P["work"] / "models" / a.baseline / "oof.parquet")
    band = oof.filter((pl.col("p") >= a.lo) & (pl.col("p") <= a.hi))
    if band.height > a.max_pairs:
        band = band.sample(a.max_pairs, seed=a.seed)
    print(f"uncertain band: {band.height} pairs in [{a.lo}, {a.hi}] out of {oof.height} "
          f"(positives {int(band['label'].sum())})", flush=True)
    if band.height == 0:
        raise SystemExit("empty uncertain band: widen --lo/--hi or check --baseline's OOF distribution")

    pq = P["parquet"] / "train"
    cols = ["rid", "core1", "addr"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=cols)
    s3 = pl.read_parquet(pq / "source3.parquet", columns=cols)
    pool = pl.concat([s2, s3])

    q = band["q"].to_numpy()
    pid = band["pid"].to_numpy()
    a_text = (s1[q]["core1"] + " " + s1[q]["addr"]).to_list()
    b_text = _pool_text(pool, pid, s2.height)
    labels = band["label"].to_numpy().astype(np.float32)

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
        dataloader_num_workers=0,  # avoid multiprocess DataLoader hang under nohup/no-tty
        seed=a.seed,
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
    trainer.train()
    model.save_pretrained(str(out))

    cfg = {"base_model": a.base_model, "base_model_license": BASE_MODEL_LICENSE,
           "base_model_params_approx": BASE_MODEL_PARAMS_APPROX, "baseline": a.baseline,
           "lo": a.lo, "hi": a.hi, "n_pairs": band.height, "epochs": a.epochs, "seed": a.seed}
    (P["work"] / "models" / a.name).mkdir(parents=True, exist_ok=True)
    (P["work"] / "models" / a.name / "xenc_config.json").write_text(json.dumps(cfg, indent=2))
    log_stage("train_xenc", cfg, {"pairs": float(band.height), "seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s, saved to {out}", flush=True)


if __name__ == "__main__":
    main()
