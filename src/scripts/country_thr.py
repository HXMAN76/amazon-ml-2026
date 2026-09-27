"""Per-country decision thresholds for the labelled countries. Usage: python src/scripts/country_thr.py MODEL
Tunes one threshold per country (US, India) on MODEL's out-of-fold tuning pairs (models/MODEL/oof_tune.parquet, exclusive assignment, macro F0.5
over those S1), then compares on the locked holdout: the model's single threshold against the per-country thresholds (paired bootstrap).
A gain whose interval excludes 0 is worth applying to the test output for US and India (France keeps its own decoding)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr0 = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    ctry = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "ctry")
    oof = pl.read_parquet(mdl / "oof_tune.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    hold = pl.read_parquet(mdl / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    grid = np.round(np.arange(0.50, 0.96, 0.02), 2)
    best = {}
    for c in ("us", "india"):
        qs = ctry.filter(pl.col("ctry") == c).select("q")
        o = oof.join(qs, on="q", how="semi")
        nt = o.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
        ex = decision.assign_exclusive(o)
        scores = [(t, decision.macro_f05(ex.filter(pl.col("p") >= t), nt)) for t in grid]
        best[c] = max(scores, key=lambda x: x[1])[0]
        print(f"{c}: out-of-fold S1 {nt.height}; F0.5 at {thr0:.2f}: {decision.macro_f05(ex.filter(pl.col('p') >= thr0), nt):.5f}; best {best[c]:.2f}: {max(s for _, s in scores):.5f}", flush=True)
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    nt = hq.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    ex = decision.assign_exclusive(hold.join(hq, on="q", how="semi")).join(ctry, on="q", how="left")
    base = decision.per_entity_f05(ex.filter(pl.col("p") >= thr0), nt).sort("q")
    thr_c = pl.col("ctry").replace_strict(best, default=thr0, return_dtype=pl.Float64)
    new = decision.per_entity_f05(ex.filter(pl.col("p") >= thr_c), nt).sort("q")
    d, lo, hi = decision.paired_bootstrap_delta(base["f"].to_numpy(), new["f"].to_numpy())
    print(json.dumps({"model": name, "threshold": thr0, "per_country": best, "holdout_base": float(base["f"].mean()), "holdout_per_country": float(new["f"].mean()),
                      "delta": d, "delta_ci95": [lo, hi]}, indent=2))


if __name__ == "__main__":
    main()
