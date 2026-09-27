"""Raw examples of France pairs from one cell of excess_mass.py. Usage: python src/scripts/cell_samples.py MODEL
Cell: name similarity >= 95, address similarity 60 to 90, same house number (the largest excess of France over the US), and the same for US as reference."""

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
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "addr", "ctry"]).rename({"rid": "q", "name1": "a_name", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"name1": "b_name", "addr": "b_addr"})
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "addr_tset", "house_eq", "addr_lev")
              .join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    d = d.join(feat, on=["q", "pid"], how="left").filter((pl.col("name_tset") >= 95) & (pl.col("addr_tset") >= 60) & (pl.col("addr_tset") < 90) & (pl.col("house_eq") > 0.5))
    for c, n in (("france", 40), ("us", 20)):
        x = d.filter(pl.col("ctry") == c)
        print(f"\n== {c}: {x.height} pairs in the cell; mean p {float(x['p'].mean()):.4f}")
        with pl.Config(tbl_rows=45, fmt_str_lengths=70, tbl_width_chars=260):
            print(x.sample(n, seed=7).select("p", "a_addr", "b_addr"))


if __name__ == "__main__":
    main()
