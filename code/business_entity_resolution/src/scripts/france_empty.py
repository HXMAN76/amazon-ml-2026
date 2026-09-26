"""Which S1 does a decoding rule leave with an empty list? Usage: python src/scripts/france_empty.py BASE VARIANT
The metric is per S1: an S1 that has true matches but gets an empty list scores 0, while one extra wrong pair on an S1 with three true
matches costs only about 0.1. The France rules judge pairs one by one (drop a category when more than 26% of it is wrong), so this counts, per
country, the S1 that BASE predicts something for and VARIANT leaves empty, with the probability of their best BASE pair and whether that
pair is an exact-name copy. Compares the share of empty S1 with the training singleton share (5.6%)."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def main() -> None:
    base, var = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / base / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"))
                      for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    own = {}
    for n in (base, var):
        pp = pl.read_parquet(P["work"] / "output" / n / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
        own[n] = decision.assign_exclusive(pp).filter(pl.col("p") >= thr)
    nb = own[base].group_by("q").len().rename({"len": "n_base"})
    nv = own[var].group_by("q").len().rename({"len": "n_var"})
    best = (own[base].join(pool, on="pid", how="left").join(s1.select("q", "a_core"), on="q", how="left")
            .sort("p", descending=True).group_by("q").first().select("q", pl.col("p").alias("best_p"), (pl.col("a_core") == pl.col("b_core")).alias("best_exact")))
    t = s1.join(nb, on="q", how="left").join(nv, on="q", how="left").join(best, on="q", how="left").with_columns(pl.col("n_base").fill_null(0), pl.col("n_var").fill_null(0))
    summ = t.group_by("ctry").agg(pl.len().alias("S1"), (pl.col("n_base") == 0).mean().alias(f"empty_{base}"), (pl.col("n_var") == 0).mean().alias(f"empty_{var}"),
                                  ((pl.col("n_base") > 0) & (pl.col("n_var") == 0)).sum().alias("emptied"),
                                  pl.col("n_base").mean().alias(f"matches_{base}"), pl.col("n_var").mean().alias(f"matches_{var}")).sort("ctry")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220):
        print(f"per country (threshold {thr:.3f}); training singleton share 0.056\n{summ}")
        e = t.filter((pl.col("n_base") > 0) & (pl.col("n_var") == 0))
        print(f"\nS1 emptied by {var}: {e.height}")
        print(e.group_by("ctry", "n_base").len().sort("ctry", "n_base"))
        print(e.with_columns(pl.col("best_p").cut([0.9, 0.95, 0.985, 0.995, 0.999], left_closed=True).alias("best_p_zone"))
               .group_by("ctry", "best_p_zone", "best_exact").len().sort("ctry", "best_p_zone", "best_exact"))


if __name__ == "__main__":
    main()
