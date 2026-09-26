"""Pairs whose names differ by one swapped common word ("nje ecole eurl" against "nje centre eurl"): how often, and are they true?
Usage: python src/scripts/word_swap.py MODEL. A swap = the core names differ by exactly one token on each side (the rest equal as sets) and both
tokens are common (they occur in at least DF_MIN core names of the split's S1 file); such pairs are near-copies of a sibling business, not noise.
Holdout: true share of predicted pairs with a swap, by probability zone (labelled). Test: share of the predicted pairs with a swap, per country."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000
DF_MIN = 100


def tok_df(s1: pl.DataFrame) -> pl.DataFrame:
    return s1.select(pl.col("core1").str.split(" ").list.unique().alias("t")).explode("t").group_by("t").len().rename({"len": "df"})


def flag(pairs: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """pairs: (a_core, b_core). Adds swap (bool): exactly one token differs on each side and both are common."""
    d = pairs.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns(pl.col("ta").list.set_difference(pl.col("tb")).alias("da"), pl.col("tb").list.set_difference(pl.col("ta")).alias("db"))
    one = (pl.col("da").list.len() == 1) & (pl.col("db").list.len() == 1)
    d = d.with_columns(pl.when(one).then(pl.col("da").list.first()).otherwise(None).alias("xa"), pl.when(one).then(pl.col("db").list.first()).otherwise(None).alias("xb"))
    dfm = dict(zip(df["t"].to_list(), df["df"].to_list()))
    ca = pl.col("xa").replace_strict(dfm, default=0, return_dtype=pl.Int64)
    cb = pl.col("xb").replace_strict(dfm, default=0, return_dtype=pl.Int64)
    return d.with_columns((one & (ca >= DF_MIN) & (cb >= DF_MIN) & (pl.col("xa").str.len_chars() > 2) & (pl.col("xb").str.len_chars() > 2)).alias("swap")).drop("ta", "tb", "da", "db", "xa", "xb")


def zone(p: pl.Expr) -> pl.Expr:
    return (pl.when(p < 0.9).then(pl.lit("1 [thr,0.9)")).when(p < 0.99).then(pl.lit("2 [0.9,0.99)")).when(p < 0.9999).then(pl.lit("3 [0.99,0.9999)")).otherwise(pl.lit("4 >=0.9999")))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    for split in ("train", "test"):
        s1 = pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
        pool = pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
        df = tok_df(s1.rename({"a_core": "core1"}))
        if split == "train":
            hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
            ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            sel = decision.assign_exclusive(ph).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
            sel = flag(sel, df).with_columns(zone(pl.col("p")).alias("zone"))
            r = sel.group_by("zone", "swap").agg(pl.len().alias("pairs"), pl.col("label").mean().alias("true_share")).sort("zone", "swap")
            print(f"== HOLDOUT predicted pairs (labelled), swap = one common word swapped\n{r}\nshare of predicted pairs with a swap: {float(sel['swap'].mean()):.4f}")
        else:
            pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            sel = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
            sel = flag(sel, df).with_columns(zone(pl.col("p")).alias("zone"))
            r = sel.group_by("ctry", "zone").agg(pl.len().alias("pairs"), pl.col("swap").mean().alias("share_swap")).sort("ctry", "zone")
            print(f"\n== TEST predicted pairs: share with a common-word swap, by country and zone\n{r}")
            print(sel.group_by("ctry").agg(pl.len().alias("pairs"), pl.col("swap").sum().alias("n_swap"), pl.col("swap").mean().alias("share_swap")).sort("ctry"))
            fr = sel.filter((pl.col("ctry") == "france") & pl.col("swap")).sample(20, seed=5)
            with pl.Config(tbl_rows=25, fmt_str_lengths=50, tbl_width_chars=200):
                print("\nFrance swap examples\n", fr.select("p", "a_core", "b_core"))


if __name__ == "__main__":
    main()
