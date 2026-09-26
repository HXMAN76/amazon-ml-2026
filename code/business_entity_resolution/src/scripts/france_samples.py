"""Look at France (no labels): frequent names, tiny pool names, and raw examples of the predictions the model is least certain about
or that break the known limits. Usage: python src/scripts/france_samples.py MODEL"""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def show(df: pl.DataFrame, n: int = 12) -> None:
    with pl.Config(tbl_rows=n, tbl_cols=12, tbl_width_chars=230, fmt_str_lengths=60):
        print(df.head(n))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "addr", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "addr", "ctry"])
                        .with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr)
    fr1, frp = s1.filter(pl.col("ctry") == "france"), pool.filter(pl.col("ctry") == "france")
    print("== France S1 sample"); show(fr1.sample(12, seed=1).select("name1", "addr"))
    print("== France S1 most frequent core names"); show(fr1.group_by("core1").len().sort("len", descending=True), 15)
    print("== France pool most frequent core names"); show(frp.group_by("core1").len().sort("len", descending=True), 25)
    for c in ("france", "us", "india"):
        pc = pool.filter(pl.col("ctry") == c)
        ln = pc["core1"].str.len_chars()
        print(f"== {c}: pool core name length <=2: {float((ln <= 2).mean()):.4f}, <=3: {float((ln <= 3).mean()):.4f}, <=5: {float((ln <= 5).mean()):.4f}; S1 core length <=5: {float((s1.filter(pl.col('ctry') == c)['core1'].str.len_chars() <= 5).mean()):.4f}")
    j = own.join(s1.rename({"name1": "s1_name", "core1": "s1_core", "addr": "s1_addr", "ctry": "s1_ctry"}), on="q", how="left").join(
        pool.rename({"name1": "b_name", "core1": "b_core", "addr": "b_addr"}).drop("ctry"), on="pid", how="left")
    f = j.filter(pl.col("s1_ctry") == "france")
    print("== France predicted pairs with p in [0.9, 0.99)"); show(f.filter((pl.col("p") >= 0.9) & (pl.col("p") < 0.99)).sample(15, seed=2).select("p", "s1_name", "s1_addr", "b_name", "b_addr"), 15)
    print("== France predicted pairs with p >= 0.9995 (random)"); show(f.filter(pl.col("p") >= 0.9995).sample(10, seed=3).select("p", "s1_name", "s1_addr", "b_name", "b_addr"), 10)
    tiny = f.filter(pl.col("b_core").str.len_chars() <= 3)
    print(f"== France predicted pairs whose pool core name has <=3 characters: {tiny.height} of {f.height}")
    show(tiny.sample(min(12, tiny.height), seed=4).select("p", "s1_name", "s1_addr", "b_name", "b_addr"), 12)
    bad = f.group_by("q").agg((pl.col("src") == 2).sum().alias("n2")).filter(pl.col("n2") > 5)
    print(f"== France S1 with more than 5 predicted S2: {bad.height}; one example")
    ex = bad.head(3)["q"].to_list()
    for q in ex:
        show(f.filter(pl.col("q") == q).sort("src", "p").select("p", "src", "s1_name", "s1_addr", "b_name", "b_addr"), 14)
    same_addr = f.with_columns((pl.col("s1_addr") == pl.col("b_addr")).alias("addr_eq"), (pl.col("s1_core") == pl.col("b_core")).alias("core_eq"))
    print("== France predicted pairs: share with exactly equal address / equal core name / both / neither")
    print(same_addr.group_by("addr_eq", "core_eq").len().with_columns((pl.col("len") / f.height).alias("share")).sort("addr_eq", "core_eq"))
    j2 = j.filter(pl.col("s1_ctry") == "us").with_columns((pl.col("s1_addr") == pl.col("b_addr")).alias("addr_eq"), (pl.col("s1_core") == pl.col("b_core")).alias("core_eq"))
    print("== US predicted pairs (same breakdown)")
    print(j2.group_by("addr_eq", "core_eq").len().with_columns((pl.col("len") / j2.height).alias("share")).sort("addr_eq", "core_eq"))


if __name__ == "__main__":
    main()
