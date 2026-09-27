"""Raw examples of France pairs the threshold rule drops: non-exact names, probability between the model threshold and 0.985. Usage: python src/scripts/lowp_samples.py MODEL"""

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
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "addr", "ctry"]).rename({"rid": "q", "name1": "a_name", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"name1": "b_name", "core1": "b_core", "addr": "b_addr"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter((pl.col("p") >= thr) & (pl.col("p") < 0.985)).join(s1, on="q", how="left").join(pool, on="pid", how="left").filter(pl.col("ctry") == "france")
    d = flag(d, df).filter((pl.col("a_core") != pl.col("b_core")) & ~pl.col("swap"))
    print(f"France non-exact non-swap pairs with p in [{thr:.2f}, 0.985): {d.height}")
    with pl.Config(tbl_rows=60, fmt_str_lengths=44, tbl_width_chars=250):
        print(d.sample(50, seed=11).select("p", "a_name", "b_name", "a_addr", "b_addr"))


if __name__ == "__main__":
    main()
