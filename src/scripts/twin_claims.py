"""Do several S1 claim the same pool record with high probability (twins)? Usage: python src/scripts/twin_claims.py MODEL [test|train]
Per country: share of pool records with at least two S1 claiming them at p >= 0.9 (and 0.5), the share of the exclusively assigned pairs that
lost such a tie, S1 that end with no match although a candidate reached p >= 0.9 for them (starved), and the pairs an S1 would gain if every
claimant kept every pool record it claims at p >= 0.9 ('claimed by all')."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    if split == "test":
        pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet")
    else:
        pp = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet")
    pp = pp.select("q", "pid", "p").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    s1 = pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pp = pp.join(s1, on="q", how="left")
    strong = pp.filter(pl.col("p") >= thr)
    for c in [x for x in ("france", "us", "india") if x in s1["ctry"].unique().to_list()]:
        d = strong.filter(pl.col("ctry") == c)
        by = d.group_by("pid").agg(pl.len().alias("k"), pl.col("p").min().alias("pmin"), (pl.col("p").max() - pl.col("p").min()).alias("gap"))
        n_pid = by.height
        multi = by.filter(pl.col("k") >= 2)
        tie = by.filter((pl.col("k") >= 2) & (pl.col("gap") < 0.01))
        excess = int((by["k"] - 1).sum())
        own = decision.assign_exclusive(pp.filter(pl.col("ctry") == c)).filter(pl.col("p") >= thr)
        served = set(own["q"].unique().to_list())
        cands = set(d["q"].unique().to_list())
        starved = len(cands - served)
        n_s1 = s1.filter(pl.col("ctry") == c).height
        print(f"\n== {c}: pool records with a claimant at p>={thr:.2f}: {n_pid}; with two or more claimants {multi.height} ({multi.height / n_pid:.4f}), "
              f"near-ties (claimant probabilities within 0.01) {tie.height} ({tie.height / n_pid:.4f}); extra claims {excess} ({excess / max(own.height, 1):.4f} of the exclusive pairs); "
              f"S1 with a claim but no exclusive match (starved) {starved} ({starved / n_s1:.4f} of S1); exclusive pairs {own.height}")
        for lvl in (0.9, 0.5):
            b2 = pp.filter((pl.col("ctry") == c) & (pl.col("p") >= lvl)).group_by("pid").len()
            print(f"   share of pool records with >=2 claimants at p>={lvl}: {float((b2['len'] >= 2).mean()):.4f} of {b2.height}")


if __name__ == "__main__":
    main()
