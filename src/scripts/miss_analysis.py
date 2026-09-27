"""Where do the blocking misses of the locked holdout come from? Categorises true pairs that were never proposed.

Usage: python src/scripts/miss_analysis.py
Categories: non-Latin pool name, empty pool address, domain/glued pool name, other. For non-Latin misses also reports the
rank the fine-tuned dense retriever gave the pair (so we can see how many a larger k would recover).
"""

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q

PID_BASE = 10_000_000


def main() -> None:
    P = config.paths()
    hq = pl.DataFrame({"q": holdout_q()})
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q")).select("q", "pid")
    lab = lab.join(hq, on="q", how="semi")
    cand = pl.concat([pl.read_parquet(f, columns=["q", "pid"]) for f in sorted((P["work"] / "blocks" / "train").glob("cand_*.parquet"))])
    cand = cand.join(hq, on="q", how="semi")
    miss = lab.join(cand, on=["q", "pid"], how="anti")
    print(f"true pairs {lab.height}, never proposed {miss.height} ({miss.height / lab.height:.4f})")
    cols = ["rid", "core1", "addr", "nl_name", "nl_addr", "ctry"]
    s2 = pl.read_parquet(P["parquet"] / "train" / "source2.parquet", columns=cols).with_columns((pl.col("rid").cast(pl.Int64) + 2 * PID_BASE).alias("pid"))
    s3 = pl.read_parquet(P["parquet"] / "train" / "source3.parquet", columns=cols).with_columns((pl.col("rid").cast(pl.Int64) + 3 * PID_BASE).alias("pid"))
    pool = pl.concat([s2, s3]).drop("rid")
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1", "addr"]).rename(
        {"rid": "q", "core1": "a_core", "addr": "a_addr"})
    m = miss.join(pool, on="pid", how="left").join(s1, on="q", how="left")
    m = m.with_columns(
        pl.when(pl.col("nl_name") > 0).then(pl.lit("1 non-Latin pool name"))
          .when(pl.col("addr").str.len_chars() == 0).then(pl.lit("2 empty pool address"))
          .when(pl.col("core1").str.contains(r"\.(com|net|org|in|co|io)\b|^\S+$")).then(pl.lit("3 domain or single glued word"))
          .otherwise(pl.lit("4 other")).alias("cat"),
        (pl.col("a_addr").str.len_chars() == 0).alias("s1_empty_addr"))
    print(m.group_by("cat").agg(pl.len().alias("n"), pl.col("s1_empty_addr").mean().alias("s1_addr_empty")).sort("cat"))
    print(m.group_by("ctry").len().sort("len", descending=True))
    dp = P["work"] / "dense" / "train" / "pairs.parquet"
    if dp.exists():
        pairs = pl.read_parquet(dp)
        nl = m.filter(pl.col("cat") == "1 non-Latin pool name").select("q", "pid").join(pairs, on=["q", "pid"], how="left")
        r = nl["rank"].to_numpy()
        print(f"non-Latin misses {nl.height}; found by dense at all (k={pairs['rank'].max() + 1}): {np.isfinite(r.astype(float)).mean():.3f}")
        for k in (5, 10, 20):
            print(f"  dense rank < {k}: {float(np.nan_to_num(r.astype(float), nan=1e9).__lt__(k).mean()):.3f}")
    print("\nsamples per category")
    for c in sorted(m["cat"].unique()):
        print(c)
        print(m.filter(pl.col("cat") == c).sample(8, seed=1).select("a_core", "a_addr", "core1", "addr"))


if __name__ == "__main__":
    main()
