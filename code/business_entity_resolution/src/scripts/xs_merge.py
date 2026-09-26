"""Cross-encoder scores for a stack rebuild in which only France changes. Usage: python src/scripts/xs_merge.py OUT BASE_DIR FR_DIR [--keep-old-noise]
Writes WORK/OUT/test_xs.parquet = WORK/BASE_DIR/test_xs.parquet with the scores of the pairs in WORK/FR_DIR/test_xs.parquet (the France pairs, scored
by a France-aware cross-encoder) replaced. `stack build --split test --xenc-dir OUT` then gives the stacked model the new France scores and
leaves every other country's features, and so its probabilities, exactly as before.
--keep-old-noise: pairs whose names differ by one word swapped into France's noise vocabulary (fils, groupe, services, developpement: the words true
copies swap in, research.md 25 and our label-free rate analysis) keep the old score; the France-aware model learned all swaps as siblings."""

import sys

import polars as pl

from ber import config

PID_BASE = 10_000_000
NOISE_FR = ["fils", "groupe", "services", "developpement"]


def main() -> None:
    out, base, fr = sys.argv[1:4]
    W = config.paths()["work"]
    b = pl.read_parquet(W / base / "test_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    f = pl.read_parquet(W / fr / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("_new"))
    if "--keep-old-noise" in sys.argv:
        pq = W / "parquet" / "test"
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").str.split(" ").list.unique().alias("ta"))
        pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                                                                                                   pl.col("core1").str.split(" ").list.unique().alias("tb")) for s in (2, 3)])
        g = f.join(s1, on="q", how="left").join(pool, on="pid", how="left")
        noise = ((pl.col("ta").list.set_difference(pl.col("tb")).list.len() == 1) & (pl.col("tb").list.set_difference(pl.col("ta")).list.len() == 1)
                 & pl.col("tb").list.set_difference(pl.col("ta")).list.first().is_in(NOISE_FR))
        n0 = f.height
        f = g.filter(~noise).select("q", "pid", "_new")
        print(f"--keep-old-noise: {n0 - f.height} noise-word swap pairs keep the old score", flush=True)
    m = b.join(f, on=["q", "pid"], how="left")
    print(f"{b.height} pairs, {m['_new'].is_not_null().sum()} rescored; mean xs of those {m.filter(pl.col('_new').is_not_null())['xs'].mean():.4f} -> "
          f"{m['_new'].mean():.4f}", flush=True)
    (W / out).mkdir(parents=True, exist_ok=True)
    m.with_columns(pl.coalesce("_new", "xs").cast(pl.Float32).alias("xs")).drop("_new").write_parquet(W / out / "test_xs.parquet", compression="zstd")


if __name__ == "__main__":
    main()
