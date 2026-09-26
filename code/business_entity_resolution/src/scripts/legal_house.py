"""Siblings next door? Pairs whose core names agree or nearly agree but whose legal forms conflict and whose house numbers differ.
Usage: python src/scripts/legal_house.py MODEL. Among the predicted pairs (exclusive assignment, threshold): the share of such pairs in France, the US
and the labelled holdout, their true share on the holdout, and the France excess over the holdout rate (excess_mass.py logic): if the kind were as
common and as true in France as on the holdout, France would have share_hold * true_share_h * n_fr true pairs of it; the rest are wrong."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def feats(P, split: str, keys: pl.DataFrame) -> pl.DataFrame:
    dirs = ["train", "train_rest"] if split == "train" else [split]
    files = [str(f) for d in dirs for f in sorted((P["work"] / "features" / d).glob("part_*.parquet"))]
    return (pl.scan_parquet(files).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "legal_conflict", "house_eq", "name_tset")
              .join(keys.lazy(), on=["q", "pid"], how="semi").collect())


def cores(P, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                                                                                                pl.col("core1").alias("b_core")) for s in (2, 3)])
    return s1, pool


def kinds(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns((pl.col("legal_conflict") > 0.5).alias("legal_conflict"), (pl.col("house_eq") < 0.5).alias("house_differs"),
                          (pl.col("a_core") == pl.col("b_core")).alias("same_core"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    h = decision.assign_exclusive(ph).filter(pl.col("p") >= thr).join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi")
    s1, pool = cores(P, "train")
    h = kinds(h.join(feats(P, "train", h.select("q", "pid")), on=["q", "pid"], how="left").join(s1, on="q", how="left").join(pool, on="pid", how="left"))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    s1t, poolt = cores(P, "test")
    t = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1t, on="q", how="left").join(poolt, on="pid", how="left")
    t = kinds(t.join(feats(P, "test", t.select("q", "pid")), on=["q", "pid"], how="left"))
    key = ["legal_conflict", "house_differs", "same_core"]
    n_h, n_fr, n_us = h.height, t.filter(pl.col("ctry") == "france").height, t.filter(pl.col("ctry") == "us").height
    ho = h.group_by(key).agg(pl.len().alias("n_hold"), pl.col("label").mean().alias("true_share_hold"))
    fr = t.filter(pl.col("ctry") == "france").group_by(key).agg(pl.len().alias("n_fr"), pl.col("p").mean().alias("mean_p_fr"))
    us = t.filter(pl.col("ctry") == "us").group_by(key).agg(pl.len().alias("n_us"))
    j = (fr.join(us, on=key, how="full", coalesce=True).join(ho, on=key, how="full", coalesce=True).fill_null(0)
           .with_columns((pl.col("n_fr") / n_fr).alias("share_fr"), (pl.col("n_us") / n_us).alias("share_us"), (pl.col("n_hold") / n_h).alias("share_hold"))
           .with_columns((pl.col("n_fr") - pl.col("share_hold") * pl.col("true_share_hold") * n_fr).alias("fr_wrong_est"))
           .with_columns((pl.col("fr_wrong_est") / pl.col("n_fr")).clip(0, 1).alias("fr_wrong_share_est")).sort(key))
    with pl.Config(tbl_rows=20, tbl_cols=14, tbl_width_chars=220):
        print(f"predicted pairs: France {n_fr}, US {n_us}, holdout {n_h}")
        print(j)


if __name__ == "__main__":
    main()
