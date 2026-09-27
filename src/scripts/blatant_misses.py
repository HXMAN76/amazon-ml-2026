"""Pool records nobody got although an S1 has the same address and a similar name: recall holes, per country. Usage: python src/scripts/blatant_misses.py MODEL
For every pair (S1, pool record) with exactly the same normalised address (not empty) and core-name token Jaccard >= 0.5, whose pool record is not predicted for any S1:
count them by whether the pair was never a candidate, was shortlisted with p below the threshold, or was shortlisted and lost to another S1. Also the share of unassigned
pool records that have such an S1. Prints examples for France."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "addr", "ctry"]).rename({"rid": "q", "name1": "a_name", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "addr", "ctry"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"name1": "b_name", "core1": "b_core", "addr": "b_addr"})
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid").with_columns(pl.lit(True).alias("assigned"))
    assigned_pid = own.select("pid").unique().with_columns(pl.lit(True).alias("pid_assigned"))
    s1a = s1.filter(pl.col("a_addr").str.len_chars() > 0).select("q", "a_name", "a_core", "a_addr", "ctry")
    pa = pool.filter(pl.col("b_addr").str.len_chars() > 0).join(assigned_pid, on="pid", how="left").with_columns(pl.col("pid_assigned").fill_null(False)).filter(~pl.col("pid_assigned"))
    j = s1a.join(pa, left_on="a_addr", right_on="b_addr", how="inner", suffix="_b").rename({"ctry_b": "pool_ctry"}) if False else s1a.join(pa.rename({"b_addr": "a_addr"}), on="a_addr", how="inner", suffix="_b")
    j = j.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    j = j.with_columns((pl.col("ta").list.set_intersection(pl.col("tb")).list.len() / (pl.col("ta").list.len() + pl.col("tb").list.len() - pl.col("ta").list.set_intersection(pl.col("tb")).list.len())).alias("jac"))
    j = j.filter(pl.col("jac") >= 0.5).drop("ta", "tb")
    j = j.join(pp.select("q", "pid", pl.col("p").alias("p_pair")), on=["q", "pid"], how="left")
    j = j.with_columns(pl.when(pl.col("p_pair").is_null()).then(pl.lit("1 never a candidate")).otherwise(pl.lit("2 shortlisted, not assigned (p below threshold or lost)")).alias("fate"))
    r = j.group_by("ctry", "fate").agg(pl.len().alias("pairs"), pl.col("pid").n_unique().alias("pool_records")).sort("ctry", "fate")
    n_pool = pool.group_by("ctry").len().rename({"len": "pool"})
    r = r.join(n_pool, on="ctry", how="left").with_columns((pl.col("pool_records") / pl.col("pool")).alias("share_of_pool"))
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(r)
        ex = j.filter((pl.col("ctry") == "france") & (pl.col("fate").str.starts_with("1"))).sample(25, seed=3)
        with pl.Config(fmt_str_lengths=45):
            print("\nFrance: same address, similar name, never a candidate (examples)\n", ex.select("jac", "a_name", "b_name", "a_addr"))
        ex2 = j.filter((pl.col("ctry") == "france") & (pl.col("fate").str.starts_with("2"))).sample(15, seed=4)
        with pl.Config(fmt_str_lengths=45):
            print("\nFrance: shortlisted but unassigned (examples)\n", ex2.select("p_pair", "jac", "a_name", "b_name", "a_addr"))


if __name__ == "__main__":
    main()
