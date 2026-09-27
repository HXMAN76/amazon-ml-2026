"""Decoy share of France pairs by probability zone and name relation, from slot occupancy (see decoy_by_category.py). Usage: python src/scripts/decoy_by_pzone.py MODEL
Zones of p (exclusive assignment, threshold) x {exact name, swap, other}. A zone is worth dropping when its decoy share is above about 26%. The portal reading of s22t2c
(threshold 0.985 on top of the type-swap rule: +0.0016) implies that about 46% of the pairs between the model threshold and 0.985 are wrong; this fits the zones above."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000
ZONES = [(0.0, 0.9), (0.9, 0.985), (0.985, 0.995), (0.995, 0.999), (0.999, 0.9999), (0.9999, 1.01)]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    rel = pl.when(pl.col("core_eq")).then(pl.lit("exact")).when(pl.col("swap")).then(pl.lit("swap")).otherwise(pl.lit("other"))
    for c in ("france", "us"):
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        rows = []
        for r in ("exact", "swap", "other"):
            for lo, hi in ZONES:
                x = d.filter((pl.col("ctry") == c) & (rel == r) & (pl.col("p") >= lo) & (pl.col("p") < hi))
                if x.height < 300:
                    continue
                cnt = x.group_by("q", "src").len().rename({"len": "n"})
                per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
                out = [r, f"[{lo},{hi})", x.height]
                for src, cap in ((2, 5), (3, 6)):
                    g = per.filter(pl.col("src") == src).group_by("k").agg(pl.len().alias("w"), pl.col("n").mean().alias("r")).filter(pl.col("k") <= cap).sort("k")
                    k = g["k"].to_numpy().astype(float); w = g["w"].to_numpy().astype(float); rr = g["r"].to_numpy(); xx = (cap - k) / cap
                    A = np.vstack([np.ones_like(xx), xx]).T * np.sqrt(w)[:, None]
                    sol, *_ = np.linalg.lstsq(A, rr * np.sqrt(w), rcond=None)
                    D = max(sol[0], 0.0); tot = float((w * rr).sum())
                    out.append(round(min(1.0, D * w.sum() / tot), 2) if tot > 0 else float("nan"))
                rows.append(out)
        with pl.Config(tbl_rows=40, tbl_width_chars=160):
            print(f"\n== {c}: decoy share by zone (S2, S3)\n", pl.DataFrame(rows, schema=["relation", "p zone", "pairs", "decoy_S2", "decoy_S3"], orient="row"))


if __name__ == "__main__":
    main()
