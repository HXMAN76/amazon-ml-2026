"""Decision rule v0: exclusive assignment, threshold, and the competition metric (vectorised).

Vectorised macro F_0.5 over S1 queries so that thresholds can be tuned on millions of scored pairs quickly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

BETA2 = 0.25


def assign_exclusive(df: pl.DataFrame) -> pl.DataFrame:
    """Every S2/S3 record (pid) keeps only its highest-probability S1 (ties: lower q)."""
    return (df.sort(["pid", "p", "q"], descending=[False, True, False])
              .group_by("pid", maintain_order=True).first())


def per_entity_f05(pred: pl.DataFrame, n_true: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F_0.5 as a table (q, f). pred: rows (q, label) of predicted pairs; n_true: (q, n_true) for EVERY S1.

    Per S1: no prediction and no truth = 1.0; prediction without truth or truth without prediction = 0.0.
    """
    g = pred.group_by("q").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
    d = n_true.join(g, on="q", how="left").with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0))
    prec = pl.col("tp") / pl.col("n_pred").clip(lower_bound=1)
    rec = pl.col("tp") / pl.col("n_true").clip(lower_bound=1)
    f = ((1 + BETA2) * prec * rec / (BETA2 * prec + rec)).fill_nan(0.0)
    score = (pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
               .when((pl.col("n_pred") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
               .otherwise(f))
    return d.select("q", score.alias("f"))


def macro_f05(pred: pl.DataFrame, n_true: pl.DataFrame) -> float:
    """Macro F_0.5 over the S1 entities in n_true (the competition metric)."""
    return float(per_entity_f05(pred, n_true)["f"].mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval of per-entity scores (resampling S1 entities with replacement)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tune_threshold(df: pl.DataFrame, n_true: pl.DataFrame, exclusive: bool,
                   grid: np.ndarray | None = None) -> tuple[float, float, list[tuple[float, float]]]:
    """df: q, pid, p, label. Returns (best threshold, best macro F0.5, curve)."""
    grid = np.arange(0.10, 0.96, 0.01) if grid is None else grid
    base = assign_exclusive(df) if exclusive else df
    curve = []
    for t in grid:
        curve.append((float(t), macro_f05(base.filter(pl.col("p") >= t), n_true)))
    best = max(curve, key=lambda x: x[1])
    return best[0], best[1], curve
