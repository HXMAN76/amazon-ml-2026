"""Are the France swap pairs a mixture of true noisy copies and address-copying decoys? Usage: python src/scripts/swap_mixture.py MODEL
Compares, for swap pairs (see word_swap.py) and for non-swap pairs, the rates of: exact address equality, empty pool address, same legal-form
field, same first house number, pool record in S2. Holdout (US and India, labelled true pairs) gives the template of a true noisy copy; France
test pairs are split by the S1's exact-copy support (the `swap_exact` set) and by probability zone. A true copy carries independent address noise,
a decoy made by copying an S1 and changing one word does not, so an excess of exact addresses in France's swaps over the France non-swap rate
estimates the decoy share."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def rates(d: pl.DataFrame, title: str) -> None:
    r = d.group_by("grp").agg(pl.len().alias("pairs"), pl.col("addr_eq").mean().alias("addr_eq"), pl.col("b_empty").mean().alias("b_addr_empty"), pl.col("legal_eq").mean().alias("legal_eq"),
                              pl.col("house_eq").mean().alias("house_eq"), pl.col("is_s2").mean().alias("share_S2"), pl.col("p").mean().alias("mean_p")).sort("grp")
    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=200):
        print(f"\n== {title}\n{r}")


def prep(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    d = pairs.join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    exact = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q").len().rename({"len": "n_exact"})
    d = d.join(exact, on="q", how="left").with_columns(pl.col("n_exact").fill_null(0))
    h = lambda c: pl.col(c).str.extract(r"(\d+)", 1)
    return d.with_columns((pl.col("a_addr") == pl.col("b_addr")).alias("addr_eq"), (pl.col("b_addr").str.len_chars() == 0).alias("b_empty"), (pl.col("a_legal") == pl.col("b_legal")).alias("legal_eq"),
                          ((h("a_addr") == h("b_addr")) & h("a_addr").is_not_null()).alias("house_eq"), (pl.col("pid") < 3 * PID_BASE).alias("is_s2"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]

    def load(split):
        s1 = pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "core1", "addr", "legal", "ctry"]).rename({"rid": "q", "core1": "a_core", "addr": "a_addr", "legal": "a_legal"}).with_columns(pl.col("q").cast(pl.Int64))
        pool = pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "core1", "addr", "legal"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr", "legal": "b_legal"})
        return s1, pool, tok_df(s1.rename({"a_core": "core1"}))

    s1, pool, df = load("train")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = prep(decision.assign_exclusive(ph).filter(pl.col("p") >= thr), s1, pool, df).filter(pl.col("label") == 1)
    d = d.with_columns(pl.when(pl.col("swap")).then(pl.lit("holdout TRUE swap")).otherwise(pl.lit("holdout TRUE non-swap")).alias("grp"))
    rates(d, "holdout, true predicted pairs (template of a true noisy copy)")
    d2 = d.filter(pl.col("swap") & (pl.col("n_exact") >= 1)).with_columns(pl.lit("holdout TRUE swap with exact-copy support").alias("grp"))
    rates(d2, "holdout, true swap pairs whose S1 has an exact copy")

    s1, pool, df = load("test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    t = prep(decision.assign_exclusive(pp).filter(pl.col("p") >= thr), s1, pool, df)
    for c in ("france", "us"):
        x = t.filter(pl.col("ctry") == c)
        x = x.with_columns(pl.when(~pl.col("swap")).then(pl.lit("1 non-swap")).when(pl.col("n_exact") >= 1).then(pl.lit("2 swap, S1 has exact copy")).otherwise(pl.lit("3 swap, no exact copy")).alias("grp"))
        rates(x, f"test {c}, predicted pairs")
    fr = t.filter((pl.col("ctry") == "france") & pl.col("swap") & (pl.col("n_exact") >= 1))
    fr = fr.with_columns(pl.when(pl.col("p") < 0.99).then(pl.lit("a p<0.99")).when(pl.col("p") < 0.999).then(pl.lit("b [0.99,0.999)")).when(pl.col("p") < 0.9999).then(pl.lit("c [0.999,0.9999)")).otherwise(pl.lit("d >=0.9999")).alias("grp"))
    rates(fr, "test france, swap pairs with exact-copy support, by probability")
    fr = fr.with_columns(pl.col("n_exact").clip(1, 5).cast(pl.String).alias("grp"))
    rates(fr, "test france, swap pairs by number of exact copies of the S1")


if __name__ == "__main__":
    main()
