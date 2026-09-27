"""prune: learned first-stage ranker that cuts the raw blocking candidates (K=100 per S1) to the best K=30.

Blocking scores every S1 against the pool and keeps the top K by a summed-IDF score. That ranking is crude: true pairs
sit at ranks 31 to 100 for about 1% of the pairs. A tiny XGBoost model on cheap, S1-local blocking features (score,
shared tokens, per-token-type scores, rank and gap inside the S1's list) re-ranks the K_raw candidates and keeps the
best K_keep, so the expensive string features are computed on fewer pairs with higher recall.

Inputs : WORK/blocks/{split}_raw/cand_*.parquet (from `ber.stages.block --k 100 --out-name {split}_raw`)
Outputs: WORK/models/prune/{xgb.json,report.json}; WORK/blocks/{split}/cand_*.parquet (same schema plus p_block)
Usage  : python -m ber.stages.prune --train           (fit on the training sample, report recall)
         python -m ber.stages.prune --apply test     (prune a split; also `train`)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config
from ber.stages.block import PID_BASE, TYPES
from ber.tracking import log_stage

BASE_COLS = ["score", "ns", *[f"s_{t}" for t in TYPES]]


def local_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add S1-local features (computed inside each S1's candidate list, so shards split by q are self-contained)."""
    return df.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("rank_q"),
        (pl.col("score").max().over("q") - pl.col("score")).cast(pl.Float32).alias("gap_q"),
        (pl.col("score") / pl.col("score").max().over("q")).cast(pl.Float32).alias("ratio_q"),
        (pl.col("ns") - pl.col("ns").max().over("q")).cast(pl.Float32).alias("ns_gap_q"),
        pl.sum_horizontal(*[(pl.col(f"s_{t}") > 0).cast(pl.Float32) for t in TYPES]).alias("n_types"),
    ).with_columns([pl.col(c).cast(pl.Float32) for c in BASE_COLS])


def feature_names() -> list[str]:
    """Column order used by the pruner model."""
    return [*BASE_COLS, "rank_q", "gap_q", "ratio_q", "ns_gap_q", "n_types"]


def _shards(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("cand_*.parquet"))


def train(k_keep: int | None = None, params: dict | None = None) -> dict:
    """Fit the pruner on the training sample; report recall of the pruned list vs the top-K by blocking score."""
    P = config.paths()
    prm = params or config.load()["prune"]
    keep = k_keep or prm["k_keep"]
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid", "fold"]).rename({"rid": "q"})
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
        pl.col("s1_rid").alias("q"), pl.lit(1, dtype=pl.Int8).alias("label")).select("q", "pid", "label")
    parts = []
    for f in _shards(P["work"] / "blocks" / "train_raw"):
        d = pl.read_parquet(f).join(smp, on="q", how="inner")
        if d.height:
            parts.append(local_features(d))
    df = pl.concat(parts).join(lab, on=["q", "pid"], how="left").with_columns(pl.col("label").fill_null(0))
    feats = feature_names()
    x, y, fold = df.select(feats).to_numpy(), df["label"].to_numpy(), df["fold"].to_numpy()
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "max_depth": prm["max_depth"],
         "eta": prm["eta"], "seed": prm["seed"]}
    try:
        p["device"] = "cuda"
        model = xgb.train(p, xgb.DMatrix(x[fold != 0], label=y[fold != 0], feature_names=feats), prm["rounds"])
    except Exception:  # noqa: BLE001 - no GPU
        p["device"] = "cpu"
        model = xgb.train(p, xgb.DMatrix(x[fold != 0], label=y[fold != 0], feature_names=feats), prm["rounds"])
    hold = df.filter(pl.col("fold") == 0).with_columns(
        pl.Series("p_block", model.predict(xgb.DMatrix(x[fold == 0], feature_names=feats))))
    n_true = int(hold["label"].sum())
    rec_raw = n_true / max(int(lab.join(hold.select("q").unique(), on="q", how="semi").height), 1)
    by_score = hold.filter(pl.col("rank_q") <= keep)["label"].sum() / max(n_true, 1)
    by_model = hold.with_columns(pl.col("p_block").rank("ordinal", descending=True).over("q").alias("r")).filter(
        pl.col("r") <= keep)["label"].sum() / max(n_true, 1)
    rep = {"pair_recall_raw": rec_raw, f"share_of_found_kept_by_score_top{keep}": float(by_score),
           f"share_of_found_kept_by_model_top{keep}": float(by_model),
           f"pair_recall_by_score_top{keep}": float(by_score) * rec_raw, f"pair_recall_by_model_top{keep}": float(by_model) * rec_raw,
           "rows": df.height, "positives": int(y.sum())}
    out = P["work"] / "models" / "prune"
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "xgb.json"))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print("pruner report (fold 0 held out):", json.dumps(rep, indent=2), flush=True)
    return rep


def apply(split: str, k_keep: int | None = None, params: dict | None = None) -> int:
    """Prune every shard of `blocks/{split}_raw` to the best K_keep per S1 and write `blocks/{split}`."""
    P = config.paths()
    prm = params or config.load()["prune"]
    keep = k_keep or prm["k_keep"]
    model = xgb.Booster()
    model.load_model(str(P["work"] / "models" / "prune" / "xgb.json"))
    feats = feature_names()
    out = P["work"] / "blocks" / split
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    total = 0
    for f in _shards(P["work"] / "blocks" / f"{split}_raw"):
        t = time.time()
        d = local_features(pl.read_parquet(f))
        d = d.with_columns(pl.Series("p_block", model.predict(xgb.DMatrix(d.select(feats).to_numpy(), feature_names=feats))))
        d = (d.with_columns(pl.col("p_block").rank("ordinal", descending=True).over("q").alias("_r"))
              .filter(pl.col("_r") <= keep).drop("_r", "rank_q", "gap_q", "ratio_q", "ns_gap_q", "n_types")
              .with_columns(pl.col("score").cast(pl.Float64)).sort("q", "pid"))
        d.write_parquet(out / f.name, compression="zstd")
        total += d.height
        print(f"{split} {f.name}: {d.height} pairs kept in {time.time() - t:.0f}s", flush=True)
    return total


def main(argv: list[str] | None = None) -> None:
    """CLI: fit the pruner (--train) and/or apply it to splits (--apply train test)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--apply", nargs="*", default=[], choices=["train", "test"])
    a = ap.parse_args(argv)
    t0 = time.time()
    metrics = {}
    if a.train:
        metrics.update({k: float(v) for k, v in train().items() if isinstance(v, (int, float))})
    for sp in a.apply:
        metrics[f"kept_{sp}"] = float(apply(sp))
    log_stage("prune", config.load()["prune"], {**metrics, "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
