"""Why are name-only (empty pool address) true pairs missed? Locked holdout, one stacked model.

Usage: python src/scripts/attrition2.py MODEL
Prints (1) whether the row order of the three source files carries any signal (position of an S1 versus position of its matches);
(2) for true pairs whose pool address is empty: found rate by how many S1 records share the S1's core name (irreducible ambiguity) and by
whether the pool name equals the S1 name; (3) found rate by the number of other matches the S1 already has (slot evidence).
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    lab_all = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").cast(pl.Int64).alias("q"))
    n1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"]).height
    for s in (2, 3):
        d = lab_all.filter(pl.col("src") == s)
        ns = pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=["rid"]).height
        x = d["q"].to_numpy() / n1
        y = d["other_rid"].to_numpy() / ns
        print(f"row order S1 vs S{s}: pearson {np.corrcoef(x, y)[0, 1]:.5f}, share within 1% of the same relative position {float((np.abs(x - y) < 0.01).mean()):.4f} (chance 0.02)")

    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    lab = lab_all.select("q", "pid", "src").join(hq, on="q", how="semi")
    pred = pl.read_parquet(mdl / "holdout_pred.parquet").select("q", "pid", "p", "label").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    found = decision.assign_exclusive(pred).filter(pl.col("p") >= thr).select("q", "pid").with_columns(pl.lit(1).alias("found"))
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1"]).rename({"rid": "q", "core1": "s1_core"}).with_columns(pl.col("q").cast(pl.Int64))
    dup = s1.group_by("s1_core").len().rename({"len": "s1_same_core"})
    pool = pl.concat([pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=["rid", "core1", "addr"])
                        .with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid")
    pool_empty = pool.filter(pl.col("addr").str.len_chars() == 0).group_by("core1").len().rename({"len": "pool_empty_same_core"})
    t = (lab.join(found, on=["q", "pid"], how="left").join(pool, on="pid", how="left").join(s1, on="q", how="left")
            .join(dup, on="s1_core", how="left").join(pool_empty, on="core1", how="left")
            .with_columns((pl.col("addr").str.len_chars() == 0).alias("empty"), pl.col("found").is_not_null().alias("ok"),
                          (pl.col("core1") == pl.col("s1_core")).alias("same_name"), pl.col("pool_empty_same_core").fill_null(0)))
    # slot evidence: other found matches of the S1, by source
    fq = found.join(lab, on=["q", "pid"], how="inner").group_by("q", "src").len()
    n_s2 = fq.filter(pl.col("src") == 2).select("q", pl.col("len").alias("found_s2"))
    n_s3 = fq.filter(pl.col("src") == 3).select("q", pl.col("len").alias("found_s3"))
    t = t.join(n_s2, on="q", how="left").join(n_s3, on="q", how="left").with_columns(pl.col("found_s2").fill_null(0), pl.col("found_s3").fill_null(0))
    e = t.filter(pl.col("empty"))
    print(f"\nempty-address true pairs {e.height}, found {float(e['ok'].mean()):.4f}")

    def bucket(c, cuts):
        return pl.col(c).cut(cuts).alias("b")
    for col, cuts in (("s1_same_core", [1, 3, 10, 50]), ("pool_empty_same_core", [0, 1, 3, 10, 50])):
        print(f"\nempty-address pairs by {col}")
        print(e.with_columns(bucket(col, cuts)).group_by("b").agg(pl.len().alias("pairs"), pl.col("ok").mean().alias("found")).sort("b"))
    print("\nempty-address pairs by same name as the S1")
    print(e.group_by("same_name").agg(pl.len().alias("pairs"), pl.col("ok").mean().alias("found")))
    print("\nnon-empty pairs by same name as the S1")
    print(t.filter(~pl.col("empty")).group_by("same_name").agg(pl.len().alias("pairs"), pl.col("ok").mean().alias("found")))
    e = e.with_columns(pl.when(pl.col("src") == 2).then(pl.col("found_s2")).otherwise(pl.col("found_s3")).alias("same_src_found"),
                       pl.when(pl.col("src") == 2).then(pl.col("found_s3")).otherwise(pl.col("found_s2")).alias("other_src_found"))
    print("\nempty-address pairs by number of the S1's other found matches in the same source (excluding itself when found)")
    print(e.with_columns(pl.col("same_src_found").clip(0, 4)).group_by("same_src_found").agg(pl.len().alias("pairs"), pl.col("ok").mean().alias("found")).sort("same_src_found"))
    print("\nempty-address pairs by the S1's found matches in the other source")
    print(e.with_columns(pl.col("other_src_found").clip(0, 4)).group_by("other_src_found").agg(pl.len().alias("pairs"), pl.col("ok").mean().alias("found")).sort("other_src_found"))


if __name__ == "__main__":
    main()
