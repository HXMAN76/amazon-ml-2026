"""The one look at the untouched final holdout (split.final_q), at the freeze.

Usage: python src/scripts/final_check.py MODEL [MODEL ...]
For each model (a stack or a blend with models/<name>/final_pred.parquet and config.json), macro F0.5 on the final holdout with the
model's own threshold, exclusive assignment and cap, its 95% interval, and the same model's locked-holdout score for comparison.
Many versions were compared on the locked holdout; if the pick is real, its final-holdout score stays inside the interval.
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import final_q


def main() -> None:
    P = config.paths()
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": final_q()}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    for name in sys.argv[1:]:
        m = P["work"] / "models" / name
        cfg = json.loads((m / "config.json").read_text())
        cap = {int(k): int(v) for k, v in (cfg.get("cap") or {}).items()}
        pred = pl.read_parquet(m / "final_pred.parquet").with_columns(pl.col("q").cast(pl.Int64))
        sel = decision.assign_exclusive(pred).filter(pl.col("p") >= cfg["threshold"])
        f = decision.per_entity_f05(decision.cap_per_source(sel, cap) if cap else sel, nt)["f"].to_numpy()
        mean, lo, hi = decision.bootstrap_ci(f)
        rep = json.loads((m / "holdout.json").read_text())
        locked = rep.get("blend_holdout_f05") if rep.get("ship") and "blend_holdout_f05" in rep else rep.get("stack_holdout_f05")
        print(f"FINAL {name}: final holdout F0.5 {mean:.5f} [{lo:.5f}, {hi:.5f}] on {len(f)} S1; locked holdout {locked}", flush=True)


if __name__ == "__main__":
    main()
