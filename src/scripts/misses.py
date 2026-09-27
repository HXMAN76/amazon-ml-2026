"""What are the true pairs a model loses on the labelled holdout? Usage: python src/scripts/misses.py MODEL
Two groups: blocking misses (true pairs never proposed: not in holdout_pred.parquet) and true candidates not kept (proposed, but below the
threshold or owned by another S1 after exclusive assignment). Each is split by kind (exact core name, same address with no common name word,
non-Latin name, domain name, some shared name word, other) with raw examples, to find a pattern a new blocking channel or rule could recover."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def kinds(d: pl.DataFrame) -> pl.DataFrame:
    ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
    return d.with_columns(
        pl.when(pl.col("a_core") == pl.col("b_core")).then(pl.lit("exact_core"))
          .when(pl.col("b_nl") | pl.col("a_nl")).then(pl.lit("non_latin"))
          .when(pl.col("b_dom") | pl.col("a_dom")).then(pl.lit("domain"))
          .when((ta.list.set_intersection(tb).list.len() == 0) & (pl.col("a_addr") == pl.col("b_addr"))).then(pl.lit("same_addr_no_common_word"))
          .when(ta.list.set_intersection(tb).list.len() == 0).then(pl.lit("no_common_word"))
          .when(pl.col("a_addr") == pl.col("b_addr")).then(pl.lit("common_word_same_addr"))
          .otherwise(pl.lit("common_word_other_addr")).alias("kind"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s28"
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    lab = (pl.read_parquet(P["parquet"] / "train" / "labels.parquet")
             .select(pl.col("s1_rid").cast(pl.Int64).alias("q"), (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid").cast(pl.Int64)).alias("pid"))
             .join(hq, on="q", how="semi"))
    hold = pl.read_parquet(mdl / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    own = decision.assign_exclusive(hold)
    kept = own.filter(pl.col("p") >= thr)
    blk = lab.join(hold, on=["q", "pid"], how="anti").with_columns(pl.lit("blocking_miss").alias("group"), pl.lit(None, pl.Float32).alias("p"))
    notk = (lab.join(hold.select("q", "pid", "p"), on=["q", "pid"], how="inner").join(kept, on=["q", "pid"], how="anti")
               .join(own.select("pid", pl.col("q").alias("owner"), pl.col("p").alias("p_owner")), on="pid", how="left")
               .with_columns(pl.when(pl.col("owner") != pl.col("q")).then(pl.lit("lost_to_other_s1")).otherwise(pl.lit("below_threshold")).alias("group"))
               .select("q", "pid", "group", pl.col("p").cast(pl.Float32)))
    cols = ["rid", "core1", "addr", "nl_name", "is_domain"]
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=cols + ["ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), (pl.col("nl_name") > 0).alias("a_nl"), pl.col("is_domain").alias("a_dom"), "ctry")
    pool = pl.concat([pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=cols).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"), (pl.col("nl_name") > 0).alias("b_nl"),
        pl.col("is_domain").alias("b_dom")) for s in (2, 3)])
    d = kinds(pl.concat([blk.select("q", "pid", "group", "p"), notk]).join(s1, on="q", how="left").join(pool, on="pid", how="left"))
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=250, fmt_str_lengths=45):
        print(f"{name}: holdout true pairs {lab.height}; by group and kind")
        print(d.group_by("group", "kind").agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p")).sort("group", "n", descending=[False, True]))
        print(d.group_by("group", "ctry").len().sort("group", "ctry"))
        for (g, k), x in d.group_by("group", "kind"):
            print(f"\n{g} / {k}: {x.height} pairs")
            print(x.sample(min(6, x.height), seed=0).select("p", "a_core", "b_core", "a_addr", "b_addr"))


if __name__ == "__main__":
    main()
