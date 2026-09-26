"""Raw examples of France's likeliest missed copies: restore candidates (france_recall.restore_candidates) of kind 'other' at the S1's address with
p in [0.3, threshold), next to the same for the US, to spot France copy patterns the model never learned. Usage: python src/scripts/missing_samples.py MODEL"""

import json
import sys

import polars as pl

from ber import config
from france_recall import feats, restore_candidates, texts

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1, pool = texts(P, "test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    c = restore_candidates(pp, thr, s1, pool).filter((pl.col("kind") == "other") & (pl.col("p") >= 0.3))
    c = c.join(feats(P, "test", c.select("q", "pid")), on=["q", "pid"], how="left").filter((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
    pq = P["parquet"] / "test"
    na = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "addr"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"), pl.col("addr").alias("a_addr"))
    nb = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "addr"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                                                                                                      pl.col("name1").alias("b_name"), pl.col("addr").alias("b_addr")) for s in (2, 3)])
    c = c.join(na, on="q", how="left").join(nb, on="pid", how="left")
    with pl.Config(tbl_rows=45, tbl_cols=8, tbl_width_chars=240, fmt_str_lengths=40):
        for ctry in ("france", "us"):
            x = c.filter(pl.col("ctry") == ctry)
            print(f"\n{ctry}: {x.height} 'other' restore candidates at the S1's address with p >= 0.3")
            print(x.sample(min(40, x.height), seed=6).select("p", "a_name", "b_name", "a_addr", "b_addr"))


if __name__ == "__main__":
    main()
