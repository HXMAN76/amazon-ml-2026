"""Do US and India want different stack thresholds? Usage: python src/scripts/country_thr.py MODEL
Locked-holdout predictions of MODEL (labelled US and India S1). Sweeps the decision threshold per country, then checks honestly: tune the thresholds on one half of the holdout S1
(global and per country), score the other half with them, and the reverse. Prints the per-country curves, the two-fold macro F0.5 with a global threshold and with per-country thresholds."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

GRID = np.round(np.arange(0.40, 0.96, 0.02), 2)


def score(own: pl.DataFrame, nt: pl.DataFrame, t: float) -> float:
    return decision.macro_f05(own.filter(pl.col("p") >= t), nt)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    thr0 = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hp = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(hp)
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = hq.join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0)).join(s1, on="q", how="left")
    half = ((nt["q"].to_numpy().astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(2)).astype(int)
    nt = nt.with_columns(pl.Series("half", half))
    own = own.join(nt.select("q", "ctry", "half"), on="q", how="left")
    print(f"model {name}, global threshold {thr0}; holdout S1 by country: {nt.group_by('ctry').len().rows()}")
    countries = [c for c in nt["ctry"].unique().to_list() if c is not None]
    for c in countries:
        ntc, oc = nt.filter(pl.col("ctry") == c), own.filter(pl.col("ctry") == c)
        cur = [(t, score(oc, ntc, t)) for t in GRID]
        best = max(cur, key=lambda x: x[1])
        print(f"{c}: best t {best[0]:.2f} F0.5 {best[1]:.5f}; at global {thr0:.2f}: {score(oc, ntc, thr0):.5f}; curve " + " ".join(f"{t:.2f}:{v:.5f}" for t, v in cur[::3]))
    tot_g = tot_c = 0.0
    n = nt.height
    for tune_half in (0, 1):
        ev = 1 - tune_half
        ntt, ntv = nt.filter(pl.col("half") == tune_half), nt.filter(pl.col("half") == ev)
        ot, ov = own.filter(pl.col("half") == tune_half), own.filter(pl.col("half") == ev)
        g_best = max(GRID, key=lambda t: score(ot, ntt, t))
        pieces = 0.0
        for c in countries:
            tc = max(GRID, key=lambda t: score(ot.filter(pl.col("ctry") == c), ntt.filter(pl.col("ctry") == c), t))
            nv = ntv.filter(pl.col("ctry") == c)
            pieces += score(ov.filter(pl.col("ctry") == c), nv, tc) * nv.height
            print(f"  tuned on half {tune_half}: {c} threshold {tc:.2f} (global {g_best:.2f})")
        tot_c += pieces
        tot_g += score(ov, ntv, g_best) * ntv.height
    print(f"two-fold macro F0.5: global threshold {tot_g / n:.5f}, per-country thresholds {tot_c / n:.5f}, difference {(tot_c - tot_g) / n:+.5f}")


if __name__ == "__main__":
    main()
