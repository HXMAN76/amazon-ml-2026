"""What do the pool records look like that nobody got? Per country, test set. Usage: python src/scripts/unassigned_profile.py MODEL
For pool records that are not predicted for any S1: share with an empty address, tiny name, a core name equal to some S1's core name of the same country (a possible owner),
and of those, the share whose namesake S1 still has free slots in that source (fewer than 5 S2 or 6 S3 predicted); best shortlist probability. In the US the unassigned
records are mostly true distractors; a country where far more of them have a namesake S1 with free slots hides missed matches."""

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
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr"})
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
    cnt = own.join(pool.select("pid", "src"), on="pid", how="left").group_by("q", "src").len().rename({"len": "n_pred"})
    free = s1.join(pl.DataFrame({"src": [2, 3]}), how="cross").join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n_pred").fill_null(0)).with_columns((pl.col("n_pred") < pl.when(pl.col("src") == 2).then(5).otherwise(6)).alias("has_free"))
    fam = free.group_by("ctry", "a_core", "src").agg(pl.len().alias("n_namesakes"), pl.col("has_free").any().alias("any_free"), pl.col("has_free").sum().alias("n_free"))
    un = pool.join(own.select("pid").unique().with_columns(pl.lit(True).alias("assigned")), on="pid", how="left").filter(pl.col("assigned").is_null()).drop("assigned")
    best = pp.group_by("pid").agg(pl.col("p").max().alias("p_best"))
    un = un.join(best, on="pid", how="left").join(fam.rename({"a_core": "b_core"}), on=["ctry", "b_core", "src"], how="left")
    un = un.with_columns(pl.col("n_namesakes").fill_null(0), pl.col("any_free").fill_null(False), pl.col("n_free").fill_null(0), pl.col("p_best").fill_null(-1.0))
    g = un.group_by("ctry").agg(pl.len().alias("unassigned"), (pl.col("b_addr").str.len_chars() == 0).mean().alias("addr_empty"), (pl.col("b_core").str.len_chars() <= 3).mean().alias("tiny_name"),
                                (pl.col("n_namesakes") > 0).mean().alias("has_namesake_S1"), pl.col("any_free").mean().alias("namesake_has_free_slot"), (pl.col("p_best") < 0).mean().alias("never_candidate"),
                                ((pl.col("p_best") >= 0.1) & (pl.col("p_best") < 0.72)).mean().alias("best_p_0.1_to_thr"), (pl.col("p_best") >= 0.72).mean().alias("best_p_ge_thr_lost")).sort("ctry")
    n_pool = pool.group_by("ctry").len().rename({"len": "pool"})
    with pl.Config(tbl_rows=10, tbl_cols=14, tbl_width_chars=220):
        print(g.join(n_pool, on="ctry").with_columns((pl.col("unassigned") / pl.col("pool")).alias("unassigned_share")))
        f = un.filter((pl.col("ctry") == "france") & pl.col("any_free") & (pl.col("n_namesakes") <= 3)).sample(20, seed=3)
        with pl.Config(fmt_str_lengths=50):
            print("\nFrance unassigned records whose namesake S1 has free slots (examples)\n", f.select("b_core", "b_addr", "p_best", "n_namesakes"))


if __name__ == "__main__":
    main()
