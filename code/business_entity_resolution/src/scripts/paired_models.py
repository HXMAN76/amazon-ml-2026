"""Paired bootstrap comparison of two stacked models on the locked holdout.

Usage: python src/scripts/paired_models.py NAME_A NAME_B
Reads WORK/models/<name>/holdout_pred.parquet and holdout.json (threshold) for both models, applies exclusive assignment and the
model's own threshold, and prints the mean difference B - A of per-S1 F0.5 with its 95% paired bootstrap interval.
"""

import json
import sys

import polars as pl

from ber import config, decision


def per_entity(name: str, nt: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    mdl = config.paths()["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    pred = pl.read_parquet(mdl / "holdout_pred.parquet")
    f = decision.per_entity_f05(decision.assign_exclusive(pred).filter(pl.col("p") >= thr), nt).sort("q")
    return f, thr


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    P = config.paths()
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    from ber.split import holdout_q

    nt = pl.DataFrame({"q": holdout_q()}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    fa, ta = per_entity(a, nt)
    fb, tb = per_entity(b, nt)
    ea, eb = fa["f"].to_numpy(), fb["f"].to_numpy()
    delta, lo, hi = decision.paired_bootstrap_delta(ea, eb)
    ma, la, ha = decision.bootstrap_ci(ea)
    mb, lb, hb = decision.bootstrap_ci(eb)
    print(json.dumps({"A": a, "B": b, "n_s1": len(ea), "A_f05": ma, "A_ci95": [la, ha], "A_threshold": ta,
                      "B_f05": mb, "B_ci95": [lb, hb], "B_threshold": tb, "delta_B_minus_A": delta, "delta_ci95": [lo, hi],
                      "B_better_share": float((eb > ea).mean()), "A_better_share": float((ea > eb).mean())}, indent=2))


if __name__ == "__main__":
    main()
