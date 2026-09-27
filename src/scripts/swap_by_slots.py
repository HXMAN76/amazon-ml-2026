"""Do swap pairs compete for the S1's slots (true copies) or are they independent of how full the S1 is (decoys)? Usage: python src/scripts/swap_by_slots.py MODEL
Per source (S2 cap 5, S3 cap 6) and per number k of the S1's confident exact-name copies in that source: number of S1, swap pairs per S1, and other
non-exact pairs per S1. True copies must fit into the remaining slots, so their rate falls to zero at k = cap; decoys do not care. France and US side by side."""

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
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    allq = s1.join(pl.DataFrame({"src": [2, 3]}), how="cross")
    per = allq.join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    sw = d.filter(pl.col("swap")).group_by("q", "src").len().rename({"len": "n_swap"})
    other = d.filter(~pl.col("core_eq") & ~pl.col("swap")).group_by("q", "src").len().rename({"len": "n_other"})
    per = per.join(sw, on=["q", "src"], how="left").join(other, on=["q", "src"], how="left").with_columns(pl.col("n_swap").fill_null(0), pl.col("n_other").fill_null(0))
    for c in ("france", "us"):
        r = per.filter(pl.col("ctry") == c).group_by("src", "k").agg(pl.len().alias("s1"), pl.col("n_swap").mean().alias("swaps_per_s1"), pl.col("n_other").mean().alias("other_nonexact_per_s1"),
                                                                    (pl.col("n_swap") > 0).mean().alias("share_s1_with_swap")).sort("src", "k")
        with pl.Config(tbl_rows=30, tbl_width_chars=200):
            print(f"\n== {c}\n{r}")


if __name__ == "__main__":
    main()
