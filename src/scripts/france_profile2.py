"""Shape of the predictions per country on the test set: probability quantiles, and the counts per S1 against the known limits (at most 5 S2
and 6 S3 matches per S1 in the training data). Usage: python src/scripts/france_profile2.py MODEL"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp)
    s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pred = own.filter(pl.col("p") >= thr).join(s1, on="q", how="left").with_columns((pl.col("pid") < 3 * PID_BASE).alias("is_s2"))
    for c in ("france", "us", "india"):
        d = pred.filter(pl.col("ctry") == c)
        p = d["p"].to_numpy()
        qs = np.quantile(p, [0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
        print(f"\n== {c}: {d.height} predicted pairs; p quantiles 1/2/5/10/25/50%: {np.round(qs, 4).tolist()}; share p<0.9 {float((p < 0.9).mean()):.4f}, <0.95 {float((p < 0.95).mean()):.4f}, "
              f"<0.99 {float((p < 0.99).mean()):.4f}, <0.999 {float((p < 0.999).mean()):.4f}")
        per = d.group_by("q").agg(pl.col("is_s2").sum().alias("n2"), (~pl.col("is_s2")).sum().alias("n3"), pl.len().alias("n"))
        n_s1 = s1.filter(pl.col("ctry") == c).height
        print(f"   S1 {n_s1}; S1 with a match {per.height / n_s1:.4f}; mean S2 {per['n2'].sum() / n_s1:.3f} mean S3 {per['n3'].sum() / n_s1:.3f}; "
              f"S1 with more than 5 S2: {int((per['n2'] > 5).sum())}, more than 6 S3: {int((per['n3'] > 6).sum())}; matches per S1 histogram (1..8+): "
              f"{[int((per['n'] == k).sum()) for k in range(1, 8)] + [int((per['n'] >= 8).sum())]}")
        # pool side: owned share
        pool_n = {"france": 1434993, "us": 3817031, "india": 4717565}[c]
        print(f"   predicted-owned share of the country's pool: {d.height / pool_n:.4f}")


if __name__ == "__main__":
    main()
