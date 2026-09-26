"""What do the S1 look like that a France rule leaves empty? Usage: python src/scripts/emptied_samples.py BASE VARIANT
For France S1 that BASE predicts something for and VARIANT leaves empty: their best BASE pair by name relation (exact or not), address similarity
and house number (first-stage features), and how many France S1 share the S1's core name (a namesake at another address can own an
exact-name pool record); then raw examples. An exact-name pair at the same address is almost surely a true copy that the rule should keep."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def main() -> None:
    base, var = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / base / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "addr", "ctry"]).rename(
        {"rid": "q", "name1": "a_name", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = s1.join(s1.group_by("ctry", "a_core").len().rename({"len": "namesakes"}), on=["ctry", "a_core"], how="left")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"))
                      for s in (2, 3)]).drop("rid").rename({"name1": "b_name", "core1": "b_core", "addr": "b_addr"})
    own = {}
    for n in (base, var):
        pp = pl.read_parquet(P["work"] / "output" / n / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
        own[n] = decision.assign_exclusive(pp).filter(pl.col("p") >= thr)
    keep = own[var].select("q").unique()
    e = (own[base].join(keep, on="q", how="anti").sort("p", descending=True).group_by("q").first()
         .join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left"))
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "addr_tset", "house_eq")
              .join(e.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    e = e.join(feat, on=["q", "pid"], how="left").with_columns(
        (pl.col("a_core") == pl.col("b_core")).alias("exact"),
        pl.col("addr_tset").cut([60, 90], labels=["a<60", "a60-90", "a90+"], left_closed=True).cast(pl.String).alias("addr_b"),
        pl.col("namesakes").cut([1, 5], labels=["unique", "2-5", "6+"], left_closed=False).cast(pl.String).alias("names_b"))
    with pl.Config(tbl_rows=40, tbl_cols=10, tbl_width_chars=240, fmt_str_lengths=40):
        print(f"France S1 emptied by {var}: {e.height}")
        print(e.group_by("exact", "addr_b", "house_eq").agg(pl.len(), pl.col("p").mean().alias("mean_p")).sort("exact", "addr_b", "house_eq"))
        print(e.group_by("exact", "names_b").len().sort("exact", "names_b"))
        for ex in (True, False):
            print(f"\nexamples, exact={ex}")
            print(e.filter(pl.col("exact") == ex).sample(min(20, e.filter(pl.col("exact") == ex).height), seed=5)
                   .select("p", "a_name", "b_name", "a_addr", "b_addr", "namesakes"))


if __name__ == "__main__":
    main()
