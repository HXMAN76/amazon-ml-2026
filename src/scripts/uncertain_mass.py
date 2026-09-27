"""How much probability mass sits in the uncertain zone below the threshold, per country? Usage: python src/scripts/uncertain_mass.py MODEL
Shortlisted pairs per S1 and expected true pairs per S1 (sum of p) in probability bins, for the test set (all countries) and for the holdout
(US and India, where labels give the true share). A country with far more pairs in 0.1 to 0.7 than the holdout has recall that the
threshold cannot reach: true pairs that look unlikely to the model (for example because of an unseen kind of address noise)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q

BINS = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]


def table(pp: pl.DataFrame, n_s1: dict[str, int]) -> None:
    lab = [f"[{BINS[i]:g},{BINS[i + 1]:g})" for i in range(len(BINS) - 1)]
    for c in ("france", "us", "india"):
        d = pp.filter(pl.col("ctry") == c)
        if d.height == 0:
            continue
        cut = d.with_columns(pl.col("p").cut(BINS[1:-1], labels=lab, left_closed=True).cast(pl.String).alias("bin"))
        agg = cut.group_by("bin").agg(pl.len().alias("n"), pl.col("p").sum().alias("sum_p")).with_columns((pl.col("n") / n_s1[c]).alias("pairs_per_s1"), (pl.col("sum_p") / n_s1[c]).alias("expected_true_per_s1")).sort("bin")
        print(f"\n== {c}: {n_s1[c]} S1\n{agg}")
        top = d.group_by("q").agg(pl.col("p").max().alias("pmax"))
        share = top.with_columns(pl.col("pmax").cut([0.1, 0.3, 0.5, 0.7], left_closed=True).cast(pl.String).alias("b")).group_by("b").len().with_columns((pl.col("len") / n_s1[c]).alias("share_of_s1")).sort("b")
        print(f"   S1 by best candidate probability (S1 without any shortlisted pair not counted)\n{share}")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").select("q", "pid", "p").with_columns(pl.col("q").cast(pl.Int64)).join(te, on="q", how="left")
    print("######## TEST")
    table(pp, {c: te.filter(pl.col("ctry") == c).height for c in ("france", "us", "india")})
    tr = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)}).join(tr, on="q", how="left")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").select("q", "pid", "p", "label").with_columns(pl.col("q").cast(pl.Int64)).join(tr, on="q", how="left")
    print("\n######## HOLDOUT (with labels)")
    table(ph, {c: hq.filter(pl.col("ctry") == c).height for c in ("us", "india")})
    lab = ph.with_columns(pl.col("p").cut(BINS[1:-1], left_closed=True).cast(pl.String).alias("bin")).group_by("bin").agg(pl.len().alias("n"), pl.col("label").mean().alias("true_share")).sort("bin")
    print(f"\nholdout true share by bin (all countries)\n{lab}")


if __name__ == "__main__":
    main()
