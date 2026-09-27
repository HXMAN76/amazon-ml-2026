"""Where does France have more predicted pairs than the labelled countries, by kind of pair? Usage: python src/scripts/excess_mass.py MODEL
Cells = name-similarity bucket x legal-form conflict x house-number equality x address-similarity bucket (first-stage columns of the pair). For the
predicted pairs (exclusive assignment, threshold) of the holdout (labelled) and of the test set's France, prints per cell the share of pairs, the
excess of France over the holdout's share (in pairs), and the holdout's true share. A cell where France has many more pairs than the training
countries and the training countries' pairs of that kind are true is where decoys of a new kind would sit (assumes the same true-noise rates)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

COLS = ["name_tset", "addr_tset", "house_eq", "legal_conflict", "core_eq"]


def cells(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(
        pl.col("name_tset").cut([60, 80, 95], labels=["n<60", "n60-80", "n80-95", "n95+"], left_closed=True).cast(pl.String).alias("nb"),
        pl.col("addr_tset").cut([60, 90], labels=["a<60", "a60-90", "a90+"], left_closed=True).cast(pl.String).alias("ab"),
        (pl.col("house_eq") > 0.5).alias("h"), (pl.col("legal_conflict") > 0.5).alias("lc"))


def feats(P, split: str, keys: pl.DataFrame) -> pl.DataFrame:
    dirs = ["train", "train_rest"] if split == "train" else [split]
    return (pl.scan_parquet(sorted(str(f) for d in dirs for f in (P["work"] / "features" / d).glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), *COLS).join(keys.lazy(), on=["q", "pid"], how="semi").collect())


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    h = decision.assign_exclusive(ph).filter(pl.col("p") >= thr)
    h = cells(h.join(feats(P, "train", h.select("q", "pid")), on=["q", "pid"], how="left"))
    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    t = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(te, on="q", how="left")
    t = cells(t.join(feats(P, "test", t.select("q", "pid")), on=["q", "pid"], how="left"))
    key = ["nb", "ab", "h", "lc"]
    n_fr = t.filter(pl.col("ctry") == "france").height
    fr = t.filter(pl.col("ctry") == "france").group_by(key).agg(pl.len().alias("n_fr"), pl.col("p").mean().alias("mean_p_fr"))
    us = t.filter(pl.col("ctry") == "us").group_by(key).agg(pl.len().alias("n_us"))
    n_us = t.filter(pl.col("ctry") == "us").height
    ho = h.group_by(key).agg(pl.len().alias("n_h"), pl.col("label").mean().alias("true_share_h"))
    n_h = h.height
    j = fr.join(us, on=key, how="full", coalesce=True).join(ho, on=key, how="full", coalesce=True).with_columns(
        pl.col("n_fr").fill_null(0), pl.col("n_us").fill_null(0), pl.col("n_h").fill_null(0)).with_columns(
        (pl.col("n_fr") / n_fr).alias("share_fr"), (pl.col("n_us") / n_us).alias("share_us"), (pl.col("n_h") / n_h).alias("share_hold")).with_columns(
        (pl.col("n_fr") - pl.col("share_hold") * n_fr).alias("excess_vs_holdout"), (pl.col("n_fr") - pl.col("share_us") * n_fr).alias("excess_vs_us"))
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=220):
        print(f"France predicted pairs {n_fr}, holdout predicted pairs {n_h}\n")
        print(j.sort("excess_vs_us", descending=True).select(*key, "n_fr", "share_fr", "share_us", "share_hold", "excess_vs_us", "excess_vs_holdout", "true_share_h", "mean_p_fr").head(20))
        print("\nsum of positive excess vs US:", float(j.filter(pl.col("excess_vs_us") > 0)["excess_vs_us"].sum()), " vs holdout:", float(j.filter(pl.col("excess_vs_holdout") > 0)["excess_vs_holdout"].sum()))


if __name__ == "__main__":
    main()
