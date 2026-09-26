"""How common are spaced or dotted legal forms ("s a r l", "l l c") in each country, and do they sit in the uncertain pairs?
Usage: python src/scripts/spaced_legal.py MODEL. Unlabelled diagnostic on the test set; also the same base rates on train."""

import sys

import polars as pl

from ber import config

PID_BASE = 10_000_000
SPACED = r"(?:^| )(?:s a r l|s a s u|s a s|e u r l|s c i|s n c|e i r l|s c o p|l l c|l l p|p l l c|i n c|l t d|p v t|p v t l t d|c o|s a)(?: |$)"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    for split in ("train", "test"):
        s1 = pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "name1", "ctry"])
        pool = pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "name1", "ctry"]).with_columns(pl.lit(s).alias("src")) for s in (2, 3)])
        for nm, d in (("S1", s1), ("pool", pool)):
            r = d.group_by("ctry").agg(pl.len().alias("n"), pl.col("name1").str.contains(SPACED).mean().alias("spaced_legal"),
                                       pl.col("name1").str.contains(r"(?:^| )(?:s a r l|s a s u|s a s|e u r l|s c i|s n c|e i r l)(?: |$)").mean().alias("fr_spaced"),
                                       pl.col("name1").str.contains(r"(?:^| )(?:l l c|l l p|p l l c|i n c|l t d|p v t)(?: |$)").mean().alias("en_spaced")).sort("ctry")
            print(f"\n== {split} {nm}\n{r}")
    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(P["parquet"] / "test" / f"source{s}.parquet", columns=["rid", "name1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(te, on="q", how="left").join(pool, on="pid", how="left")
    pp = pp.with_columns(pl.col("name1").str.contains(SPACED).alias("sp"), pl.when(pl.col("p") >= 0.9).then(pl.lit("3 p>=0.9")).when(pl.col("p") >= 0.1).then(pl.lit("2 p in [0.1,0.9)")).otherwise(pl.lit("1 p<0.1")).alias("zone"))
    print("\n== share of shortlisted pairs whose pool name has a spaced legal form, by probability zone\n",
          pp.group_by("ctry", "zone").agg(pl.len().alias("pairs"), pl.col("sp").mean().alias("share_spaced")).sort("ctry", "zone"))


if __name__ == "__main__":
    main()
