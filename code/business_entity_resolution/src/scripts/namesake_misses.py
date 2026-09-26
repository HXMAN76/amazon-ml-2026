"""Unassigned pool records with an exact-name S1 whose address is similar: likely missed matches, per country. Usage: python src/scripts/namesake_misses.py MODEL
For pool records that no S1 got: the S1 of the same country with the same core name (at most 30 namesakes) that has the largest address-token Jaccard with the record; buckets of
that Jaccard, split by whether the pair was a candidate at all (in the shortlist) and its probability. A pair with equal names and Jaccard >= 0.5 that was not taken is a
probable miss (US as the reference for how often that happens where recall is known to be good)."""

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
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "core", "addr": "b_addr"})
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("pid").unique().with_columns(pl.lit(True).alias("assigned"))
    un = pool.join(own, on="pid", how="left").filter(pl.col("assigned").is_null()).drop("assigned").filter(pl.col("b_addr").str.len_chars() > 0)
    fam = s1.group_by("ctry", "core").len().filter(pl.col("len") <= 30).select("ctry", "core")
    j = un.join(fam, on=["ctry", "core"], how="semi").join(s1.select("q", "core", "ctry", "a_addr"), on=["ctry", "core"], how="inner")
    j = j.with_columns(pl.col("a_addr").str.split(" ").list.unique().alias("ta"), pl.col("b_addr").str.split(" ").list.unique().alias("tb"))
    j = j.with_columns((pl.col("ta").list.set_intersection(pl.col("tb")).list.len() / (pl.col("ta").list.len() + pl.col("tb").list.len() - pl.col("ta").list.set_intersection(pl.col("tb")).list.len())).alias("jac"))
    best = j.sort("jac", descending=True).group_by("pid").first()
    best = best.join(pp.select("q", "pid", pl.col("p").alias("p_pair")), on=["q", "pid"], how="left")
    best = best.with_columns(pl.col("jac").cut([0.3, 0.5, 0.7], labels=["j<0.3", "j0.3-0.5", "j0.5-0.7", "j0.7+"], left_closed=True).cast(pl.String).alias("jb"),
                             pl.when(pl.col("p_pair").is_null()).then(pl.lit("never candidate")).when(pl.col("p_pair") < 0.1).then(pl.lit("p<0.1")).otherwise(pl.lit("p>=0.1")).alias("fate"))
    n_pool = pool.group_by("ctry").len().rename({"len": "pool"})
    r = best.group_by("ctry", "jb", "fate").len().join(n_pool, on="ctry").with_columns((pl.col("len") / pl.col("pool")).alias("share_of_pool")).sort("ctry", "jb", "fate")
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(r.filter(pl.col("jb").is_in(["j0.5-0.7", "j0.7+"])))
        ex = best.filter((pl.col("ctry") == "france") & (pl.col("jb") == "j0.7+") & (pl.col("fate") == "never candidate")).sample(15, seed=1)
        with pl.Config(fmt_str_lengths=55, tbl_width_chars=230):
            print("\nFrance, exact name, similar address, never a candidate (examples)\n", ex.select("core", "b_addr", "a_addr", "jac"))


if __name__ == "__main__":
    main()
