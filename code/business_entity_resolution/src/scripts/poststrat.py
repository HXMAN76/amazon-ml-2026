"""Expected test score by post-stratification: reweight the holdout's per-S1 F0.5 to the test's mix of S1 types.

Usage: python src/scripts/poststrat.py MODEL
Strata of an S1: country group (India or US; France is mapped to each in turn), how many S1 of its split share its core name (1, 2-3, 4-10,
more than 10) and its exact address (1, 2, 3-5, 6-14, 15 or more). The holdout's mean F0.5 per stratum is applied to the test's share of
each stratum. It only sees S1 attributes (not pool-side difficulty), so it is a lower bound on the explained gap. It also prints how much of
France lies in address-sharing strata that the training data hardly covers.
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q


def strata(s1: pl.DataFrame) -> pl.DataFrame:
    core = s1.group_by("core1").len().rename({"len": "core_n"})
    addr = s1.group_by("addr").len().rename({"len": "addr_n"})
    d = s1.join(core, on="core1", how="left").join(addr, on="addr", how="left")
    return d.with_columns(
        pl.col("core_n").cut([1, 3, 10], labels=["c1", "c2-3", "c4-10", "c11+"]).cast(pl.String).alias("cb"),
        pl.col("addr_n").cut([1, 2, 5, 14], labels=["a1", "a2", "a3-5", "a6-14", "a15+"]).cast(pl.String).alias("ab"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    tr = strata(pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64)))
    te = strata(pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]))
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": holdout_q().astype(np.int64)}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    pred = pl.read_parquet(mdl / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    f = decision.per_entity_f05(decision.assign_exclusive(pred).filter(pl.col("p") >= thr), nt).join(tr.select("q", "ctry", "cb", "ab"), on="q", how="left")
    print(f"holdout macro F0.5 {float(f['f'].mean()):.5f}")
    cell = f.group_by("ctry", "cb", "ab").agg(pl.len().alias("n"), pl.col("f").mean().alias("f_cell"))
    by_ctry = f.group_by("ctry").agg(pl.col("f").mean().alias("f_marg"))
    print("\nholdout F0.5 by address-sharing bucket\n", f.group_by("ab").agg(pl.len().alias("n"), pl.col("f").mean().alias("f")).sort("ab"))
    print("\nholdout F0.5 by core-name bucket\n", f.group_by("cb").agg(pl.len().alias("n"), pl.col("f").mean().alias("f")).sort("cb"))
    def est(d: pl.DataFrame, group: str) -> tuple[float, float]:
        """Mean of the holdout stratum F0.5 over the test S1 `d`, with the strata of country `group`; and the share in thin strata."""
        e = d.with_columns(pl.lit(group).alias("ctry")).drop("ctry").with_columns(pl.lit(group).alias("ctry")).join(cell, on=["ctry", "cb", "ab"], how="left").join(by_ctry, on="ctry", how="left")
        thin = float(e.select((pl.col("n").fill_null(0) < 30).mean()).item())
        return float(e.select(pl.when(pl.col("n") >= 30).then(pl.col("f_cell")).otherwise(pl.col("f_marg")).mean()).item()), thin

    tot = 0.0
    for c in ("us", "india", "france"):
        d = te.filter(pl.col("ctry") == c)
        groups = ("us", "india") if c == "france" else (c,)
        res = {g: est(d, g) for g in groups}
        for g, (v, thin) in res.items():
            print(f"test {c} with strata of {g}: estimated F0.5 {v:.5f}; share of S1 in strata with fewer than 30 holdout S1 {thin:.4f}")
        tot += float(np.mean([v for v, _ in res.values()])) * d.height / te.height
    print(f"\nestimated test macro F0.5 from S1 strata: {tot:.5f} (holdout {float(f['f'].mean()):.5f}; portal read 0.981)")
    print("\naddress-sharing buckets, share of S1 (train / test per country)")
    for nm, d in (("train", tr), ("test", te)):
        print(nm, d.group_by("ctry", "ab").len().with_columns((pl.col("len") / pl.col("len").sum().over("ctry")).alias("share")).sort("ctry", "ab").select("ctry", "ab", "share").pivot(on="ab", index="ctry", values="share"))


if __name__ == "__main__":
    main()
