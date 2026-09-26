"""Does a confident France-aware cross-encoder score mark true copies among the pairs the model left out? Usage: python src/scripts/restore_xfr.py MODEL XDIR
Restore candidates (france_recall.restore_candidates: shortlisted pairs below the decision whose pool record nobody owns) by the cross-encoder
score of WORK/XDIR ({test,train}_xs.parquet) and by address evidence: counts for France and, on the labelled holdout (US and India), the true share
of each bin. A bin whose holdout true share is well above the break-even of an added pair (about 0.74 in F0.5) is a candidate for a restore rule."""

import json
import sys

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q
from france_recall import feats, restore_candidates, texts


def binned(c: pl.DataFrame) -> pl.DataFrame:
    return c.with_columns(pl.col("xfr").cut([0.5, 0.9, 0.99], labels=["<0.5", "0.5-0.9", "0.9-0.99", ">=0.99"], left_closed=True).cast(pl.String).alias("xfr_bin"),
                          ((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90)).alias("same_addr"))


def main() -> None:
    name, xdir = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1, pool = texts(P, "test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    c = restore_candidates(pp, thr, s1, pool).filter(pl.col("ctry") == "france")
    xf = pl.read_parquet(P["work"] / xdir / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xfr"))
    c = binned(c.join(xf, on=["q", "pid"], how="inner").join(feats(P, "test", c.select("q", "pid")), on=["q", "pid"], how="left"))
    with pl.Config(tbl_rows=40, tbl_cols=10, tbl_width_chars=200):
        print(f"France restore candidates of {name} by {xdir} score")
        print(c.group_by("same_addr", "xfr_bin").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), pl.col("slot_free").mean().alias("slot_free"))
               .sort("same_addr", "xfr_bin"))
        print(c.filter(pl.col("xfr") >= 0.99).group_by("kind", "same_addr").len().sort("kind", "same_addr"))
    s1h, poolh = texts(P, "train")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    hthr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    ph = ph.join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi")
    ch = restore_candidates(ph.drop("label"), hthr, s1h, poolh).join(ph.select("q", "pid", "label"), on=["q", "pid"], how="left")
    xh = pl.read_parquet(P["work"] / xdir / "train_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xfr"))
    ch = binned(ch.join(xh, on=["q", "pid"], how="inner").join(feats(P, "train", ch.select("q", "pid")), on=["q", "pid"], how="left"))
    with pl.Config(tbl_rows=40, tbl_cols=10, tbl_width_chars=200):
        print(f"\nholdout restore candidates by {xdir} score: true share")
        print(ch.group_by("same_addr", "xfr_bin").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), pl.col("label").mean().alias("true_share"))
                .sort("same_addr", "xfr_bin"))


if __name__ == "__main__":
    main()
