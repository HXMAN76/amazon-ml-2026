"""Kinds of predicted lists per S1: empty / only swap pairs / has a confident exact-name pair / other. France against US and India (test). Usage: python src/scripts/s1_types.py MODEL
An S1 whose whole list consists of swap pairs (one common word replaced) has no confident copy of its own name: a singleton S1 with decoys is the likely reason."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core", "name1": "a_name"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "name1": "b_name"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    g = d.group_by("q").agg(pl.len().alias("n"), pl.col("swap").all().alias("all_swap"), (pl.col("core_eq") & (pl.col("p") >= 0.999)).any().alias("has_exact"), (pl.col("swap") | (pl.col("p") < 0.99)).all().alias("all_weak"),
                            pl.col("swap").sum().alias("n_swap"))
    t = s1.select("q", "ctry", "a_name").join(g, on="q", how="left").with_columns(pl.col("n").fill_null(0))
    t = t.with_columns(pl.when(pl.col("n") == 0).then(pl.lit("1 empty")).when(pl.col("all_swap")).then(pl.lit("2 only swap pairs")).when(~pl.col("has_exact")).then(pl.lit("3 no confident exact copy, other"))
                         .otherwise(pl.lit("4 has a confident exact copy")).alias("kind"))
    r = t.group_by("ctry", "kind").agg(pl.len().alias("s1"), pl.col("n").mean().alias("mean_pairs")).with_columns((pl.col("s1") / pl.col("s1").sum().over("ctry")).alias("share")).sort("ctry", "kind")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(r)
        ex = d.join(t.filter((pl.col("ctry") == "france") & (pl.col("kind") == "2 only swap pairs")).select("q"), on="q", how="semi").sort("q").head(30)
        with pl.Config(fmt_str_lengths=45):
            print("\nFrance S1 whose list is only swap pairs (examples)\n", ex.select("p", "a_name", "b_name"))


if __name__ == "__main__":
    main()
