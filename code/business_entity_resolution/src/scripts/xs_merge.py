"""Cross-encoder scores for a stack rebuild in which only France changes. Usage: python src/scripts/xs_merge.py OUT BASE_DIR FR_DIR
Writes WORK/OUT/test_xs.parquet = WORK/BASE_DIR/test_xs.parquet with the scores of the pairs in WORK/FR_DIR/test_xs.parquet (the France pairs, scored
by a France-aware cross-encoder) replaced. `stack build --split test --xenc-dir OUT` then gives the stacked model the new France scores and
leaves every other country's features, and so its probabilities, exactly as before."""

import sys

import polars as pl

from ber import config


def main() -> None:
    out, base, fr = sys.argv[1:4]
    W = config.paths()["work"]
    b = pl.read_parquet(W / base / "test_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    f = pl.read_parquet(W / fr / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("_new"))
    m = b.join(f, on=["q", "pid"], how="left")
    print(f"{b.height} pairs, {m['_new'].is_not_null().sum()} rescored; mean xs of those {m.filter(pl.col('_new').is_not_null())['xs'].mean():.4f} -> "
          f"{m['_new'].mean():.4f}", flush=True)
    (W / out).mkdir(parents=True, exist_ok=True)
    m.with_columns(pl.coalesce("_new", "xs").cast(pl.Float32).alias("xs")).drop("_new").write_parquet(W / out / "test_xs.parquet", compression="zstd")


if __name__ == "__main__":
    main()
