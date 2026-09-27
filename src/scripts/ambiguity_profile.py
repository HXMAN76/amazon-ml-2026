"""How ambiguous are the records of each split and country? A model-independent difficulty profile (train, test, per country).

Usage: python src/scripts/ambiguity_profile.py
Per split and country: share of pool records with an empty address, share of S1 whose core name is shared with other S1 (unique, 2-3,
4-10, more than 10), share of S1 addresses shared by several S1, share of pool records whose address occurs at least 5 times in the pool.
Name-ambiguous records with no pool address are the part of the recall loss no matcher can recover (see attrition2.py), so a split with more
of them is harder for reasons that do not show in the locked holdout.
"""

import polars as pl

from ber import config


def main() -> None:
    P = config.paths()
    for split in ("train", "test"):
        s1 = pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "core1", "addr", "ctry"])
        pool = pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry"]) for s in (2, 3)])
        g = s1.group_by("core1").len().rename({"len": "core_n"})
        a = s1.group_by("addr").len().rename({"len": "addr_n"})
        s1 = s1.join(g, on="core1", how="left").join(a, on="addr", how="left")
        pa = pool.filter(pl.col("addr").str.len_chars() > 0).group_by("addr").len().rename({"len": "pool_addr_n"})
        pool = pool.join(pa, on="addr", how="left").join(g.rename({"core_n": "s1_with_same_core"}), on="core1", how="left").with_columns(
            (pl.col("addr").str.len_chars() == 0).alias("empty"), pl.col("s1_with_same_core").fill_null(0))
        r1 = s1.group_by("ctry").agg(pl.len().alias("s1"), (pl.col("core_n") == 1).mean().alias("core_unique"), ((pl.col("core_n") >= 2) & (pl.col("core_n") <= 3)).mean().alias("core_2_3"),
                                     ((pl.col("core_n") >= 4) & (pl.col("core_n") <= 10)).mean().alias("core_4_10"), (pl.col("core_n") > 10).mean().alias("core_gt10"),
                                     (pl.col("addr_n") > 1).mean().alias("addr_shared")).sort("ctry")
        r2 = pool.group_by("ctry").agg(pl.len().alias("pool"), pl.col("empty").mean().alias("pool_empty_addr"), (pl.col("pool_addr_n") >= 5).mean().alias("pool_addr_ge5"),
                                       (pl.col("empty") & (pl.col("s1_with_same_core") > 1)).mean().alias("empty_and_name_shared"),
                                       (pl.col("empty") & (pl.col("s1_with_same_core") == 0)).mean().alias("empty_and_name_absent")).sort("ctry")
        print(f"\n== {split}: S1\n{r1}\n== {split}: pool (S2+S3)\n{r2}")


if __name__ == "__main__":
    main()
