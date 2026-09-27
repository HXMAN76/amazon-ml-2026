"""Pairs that are certainly wrong in France, from the known limits (at most 5 S2 and 6 S3 matches per S1): what do they look like?
Usage: python src/scripts/sure_negatives.py MODEL. An S1 that already has 5 confident exact-name S2 copies (core name equal, p >= 0.999) cannot own a 6th S2 record, so
every further predicted S2 pair of that S1 is wrong (same for 6 exact S3 copies and further S3 pairs). Compares those sure negatives with the
sure positives (exact-name pairs with p >= 0.9999) and with all France pairs on the kind of difference, and prints examples."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "addr", "legal", "ctry"]).rename({"rid": "q", "name1": "a_name", "core1": "a_core", "addr": "a_addr", "legal": "a_legal"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = s1.join(s1.group_by("a_addr").len().rename({"len": "addr_n"}), on="a_addr", how="left")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "addr", "legal"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"name1": "b_name", "core1": "b_core", "addr": "b_addr", "legal": "b_legal"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left").filter(pl.col("ctry") == "france")
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "addr_tset", "house_eq", "legal_conflict", "addr_lev")
              .join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    d = flag(d.join(feat, on=["q", "pid"], how="left"), df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "n_ex"})
    d = d.join(ex, on=["q", "src"], how="left").with_columns(pl.col("n_ex").fill_null(0))
    cap = pl.when(pl.col("src") == 2).then(5).otherwise(6)
    d = d.with_columns(pl.when(pl.col("core_eq") & (pl.col("p") >= 0.9999)).then(pl.lit("2 sure positive (exact name, p>=0.9999)"))
                         .when((pl.col("n_ex") >= cap) & ~(pl.col("core_eq") & (pl.col("p") >= 0.999))).then(pl.lit("1 sure negative (S1 already has all its slots)"))
                         .otherwise(pl.lit("3 other")).alias("grp"))
    d = d.with_columns((pl.col("legal_conflict") > 0.5).alias("lc"), (pl.col("name_tset") < 80).alias("weak_name"), (pl.col("house_eq") < 0.5).alias("house_differs"), (pl.col("addr_tset") >= 90).alias("addr_strong"),
                       (pl.col("addr_n") >= 2).alias("addr_shared_s1"), (pl.col("b_core").str.len_chars() <= 3).alias("tiny"), (pl.col("p") < 0.99).alias("p_lt_099"), (pl.col("b_addr").str.len_chars() == 0).alias("b_addr_empty"),
                       (pl.col("a_legal") != pl.col("b_legal")).alias("legal_differs"))
    cats = ["swap", "lc", "legal_differs", "weak_name", "house_differs", "addr_strong", "addr_shared_s1", "tiny", "p_lt_099", "b_addr_empty"]
    r = d.group_by("grp").agg(pl.len().alias("pairs"), *[pl.col(c).mean().alias(c) for c in cats], pl.col("p").mean().alias("mean_p")).sort("grp")
    allr = d.select(pl.lit("0 all France predicted pairs").alias("grp"), pl.len().alias("pairs"), *[pl.col(c).mean().alias(c) for c in cats], pl.col("p").mean().alias("mean_p"))
    with pl.Config(tbl_rows=10, tbl_cols=14, tbl_width_chars=250):
        print(pl.concat([allr, r]))
        neg = d.filter(pl.col("grp").str.starts_with("1"))
        print(f"\nS1 with a sure negative: {neg['q'].n_unique()}")
        with pl.Config(fmt_str_lengths=45, tbl_width_chars=250):
            print(neg.sample(min(40, neg.height), seed=1).select("p", "src", "a_name", "b_name", "a_addr", "b_addr"))
    d.filter(pl.col("grp").str.starts_with("1")).write_parquet(P["work"] / "france_sure_negatives.parquet")


if __name__ == "__main__":
    main()
