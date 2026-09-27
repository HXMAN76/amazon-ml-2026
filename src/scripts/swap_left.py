"""The one-word swaps France still predicts after a recipe, by swapped-in word. Usage: python src/scripts/swap_left.py VARIANT MODEL
VARIANT's predicted France pairs (exclusive assignment at MODEL's threshold) whose core names differ by one common word on each side: per swapped-in
word, pairs, mean p, share at slot-full S1 (k >= 3 confident exact copies) against k = 0 per S1 (the typeswap ratio; about 1 for decoys, about 0.5
for true copies), sorted by pairs. Shows which words the learned type vocabulary misses."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    var, model = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src"))
                      for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / var / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    slots = s1.filter(pl.col("ctry") == "france").select("q").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    nA, nB = slots.filter(pl.col("k") >= 3).height, slots.filter(pl.col("k") == 0).height
    sw = d.filter(pl.col("swap")).join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    sw = sw.with_columns(pl.col("b_core").str.split(" ").list.set_difference(pl.col("a_core").str.split(" ")).list.first().alias("xb"))
    g = (sw.group_by("xb").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), (pl.col("k") >= 3).sum().alias("nA"), (pl.col("k") == 0).sum().alias("nB"),
                               (pl.col("p") >= 0.995).mean().alias("share_p>=0.995"))
           .with_columns(((pl.col("nA") / nA) / (pl.col("nB") / nB + 1e-12)).alias("ratio")).sort("pairs", descending=True))
    with pl.Config(tbl_rows=45, tbl_cols=8, tbl_width_chars=180):
        print(f"{var}: France one-word swaps still predicted: {sw.height} pairs, {sw.filter(pl.col('p') >= 0.995).height} with p >= 0.995; slots A {nA}, B {nB}")
        print(g.head(40))
        print(f"words with ratio >= 0.75 and >= 20 pairs: {g.filter((pl.col('ratio') >= 0.75) & (pl.col('pairs') >= 20)).height}, pairs {g.filter((pl.col('ratio') >= 0.75) & (pl.col('pairs') >= 20))['pairs'].sum()}")


if __name__ == "__main__":
    main()
