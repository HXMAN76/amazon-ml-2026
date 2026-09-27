"""Slot-limit instrument for street-mismatch pairs. Usage: python src/scripts/street_slots.py MODEL
For France and US (test): pairs with name similarity >= 95 and the same house number whose addresses share no rare token (street_mismatch.py). Per S1 and source, the rate of such
pairs by the number k of confident exact-name copies; true copies fall to zero at k = cap, decoys stay flat. Fit as in decoy_by_category.py."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from street_mismatch import mismatch, rare_vocab

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr"})
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "house_eq").join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    d = d.join(feat, on=["q", "pid"], how="left").with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    for c in ("france", "us"):
        rare = rare_vocab(s1.filter(pl.col("ctry") == c)["a_addr"])
        x = d.filter((pl.col("ctry") == c) & (pl.col("name_tset") >= 95) & (pl.col("house_eq") > 0.5) & ~pl.col("core_eq"))
        m = mismatch(x["a_addr"].to_list(), x["b_addr"].to_list(), rare)
        x = x.with_columns(pl.Series("sm", m)).filter(pl.col("sm"))
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        cnt = x.group_by("q", "src").len().rename({"len": "n"})
        per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
        print(f"\n== {c}: {x.height} street-mismatch pairs (name >= 95, same house number, not exact core name)")
        for src, cap in ((2, 5), (3, 6)):
            g = per.filter(pl.col("src") == src).group_by("k").agg(pl.len().alias("s1"), pl.col("n").mean().alias("rate")).filter(pl.col("k") <= cap).sort("k")
            k = g["k"].to_numpy().astype(float); w = g["s1"].to_numpy().astype(float); r = g["rate"].to_numpy(); xx = (cap - k) / cap
            A = np.vstack([np.ones_like(xx), xx]).T * np.sqrt(w)[:, None]
            sol, *_ = np.linalg.lstsq(A, r * np.sqrt(w), rcond=None)
            D, T = max(sol[0], 0), max(sol[1], 0)
            print(f"S{src}: rate by k {[round(float(v), 4) for v in r]}  fit D={D:.4f} T={T:.4f} decoy share {min(1.0, D * w.sum() / float((w * r).sum())):.2f}")


if __name__ == "__main__":
    main()
