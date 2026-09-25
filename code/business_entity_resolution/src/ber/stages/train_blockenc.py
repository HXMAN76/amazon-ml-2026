"""train_blockenc: fine-tune a small bi-encoder for dense blocking (v4.md stage 1).

Base model: intfloat/multilingual-e5-small (MIT, ~118M params, 100+ languages, offline). Trained
with supervised contrastive loss (in-batch negatives via MultipleNegativesRankingLoss) on the
known true pairs from labels.parquet -- the standard recipe for a retrieval bi-encoder (Sudowoodo,
SC-Block; see ARCHITECTURE_v3_research.md sec2). Output feeds score_blockenc.py, which retrieves
top-k candidates per source-1 record and unions them with the existing token-index candidates --
this stage only produces the encoder, it makes no match/no-match decision itself.

Inputs : DATA/train/labels.parquet (s1_rid, src, other_rid: true pairs)
         WORK/parquet/train/source{1,2,3}.parquet (raw text)
Outputs: WORK/models/<name>/blockenc/ (fine-tuned bi-encoder weights)
"""

from __future__ import annotations

import argparse
import json
import time

import polars as pl

from ber import config
from ber.stages.block import PID_BASE
from ber.tracking import log_stage

BASE_MODEL = "BAAI/bge-small-en-v1.5"
BASE_MODEL_LICENSE = "MIT"
BASE_MODEL_PARAMS_APPROX = "33M"


def main(argv: list[str] | None = None) -> None:
    """CLI: fine-tune a small multilingual bi-encoder on true pairs with in-batch-negative contrastive loss."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="blockenc0")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-pairs", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()

    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
        pl.col("s1_rid").alias("q"),
    ).select("q", "pid")
    if lab.height > a.max_pairs:
        lab = lab.sample(a.max_pairs, seed=a.seed)
    print(f"training on {lab.height} true pairs", flush=True)

    pq = P["parquet"] / "train"
    cols = ["rid", "core1", "addr"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=cols).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=cols)
    s3 = pl.read_parquet(pq / "source3.parquet", columns=cols)
    pool = pl.concat([s2.with_columns((pl.col("rid").cast(pl.Int64) + 2 * PID_BASE).alias("pid")),
                       s3.with_columns((pl.col("rid").cast(pl.Int64) + 3 * PID_BASE).alias("pid"))])

    q = lab["q"].to_numpy()
    joined = lab.join(pool, on="pid", how="left")
    a_text = ("query: " + (s1[q]["core1"] + " " + s1[q]["addr"])).to_list()
    b_text = (joined["core1"].fill_null("") + " " + joined["addr"].fill_null("")).to_list()
    # bge models expect the "query: " instruction prefix on the query side only, not on passages
    # (BAAI's documented usage for retrieval) -- not a free styling choice

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(a.base_model)
    train_dataset = Dataset.from_dict({"anchor": a_text, "positive": b_text})
    loss = MultipleNegativesRankingLoss(model)
    out = P["work"] / "models" / a.name / "blockenc"
    out.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
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
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
    trainer.train()
    model.save_pretrained(str(out))

    cfg = {"base_model": a.base_model, "base_model_license": BASE_MODEL_LICENSE,
           "base_model_params_approx": BASE_MODEL_PARAMS_APPROX, "n_pairs": lab.height,
           "epochs": a.epochs, "seed": a.seed}
    (P["work"] / "models" / a.name / "blockenc_config.json").write_text(json.dumps(cfg, indent=2))
    log_stage("train_blockenc", cfg, {"pairs": float(lab.height), "seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s, saved to {out}", flush=True)


if __name__ == "__main__":
    main()
