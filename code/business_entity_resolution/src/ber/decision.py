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


def consensus_prune(sel: pl.DataFrame) -> pl.DataFrame:
    """Drop numeric look-alike distractors from an already-selected set (post `assign_exclusive` + threshold).

    84% of v0's false positives are look-alike distractors: near-copies with 1-2 changed digits (research.md
    section 13). `pairs.blocking_features`/`string_features` already compute `pin_match` (postcode agrees with
    the S1), `pin_conflict` (postcode present on both sides but disagrees) and `house_eq` (house number agrees)
    per pair; this never needed a new model or a new join, only reading what the matcher already had.

    Rule: within an S1 (`q`) whose selected members mostly confirm on postcode or house number (majority
    `pin_match | house_eq`), drop any member that actively *conflicts* (`pin_conflict` and not `house_eq`)
    rather than merely lacking a postcode. A minority member with no numeric evidence either way is left alone
    (most addresses lack a postcode/house number at all: pruning on absence would cost recall for no reason).
    Singletons (n_sel == 1) and S1 where the majority itself lacks numeric confirmation are left untouched:
    the rule only fires where siblings give a confident signal to disagree with.
    """
    need = {"q", "pid", "p", "pin_match", "pin_conflict", "house_eq"}
    missing = need - set(sel.columns)
    if missing:
        raise ValueError(f"consensus_prune needs columns {sorted(missing)}")
    g = sel.with_columns((pl.col("pin_match") | pl.col("house_eq")).alias("_confirms")).with_columns(
        pl.len().over("q").alias("_n_sel"),
        pl.col("_confirms").sum().over("q").alias("_n_confirm"),
    )
    outlier = ((pl.col("_n_sel") > 1) & (pl.col("_n_confirm") / pl.col("_n_sel") >= 0.5)
               & pl.col("pin_conflict") & ~pl.col("house_eq"))
    return g.filter(~outlier).drop("_confirms", "_n_sel", "_n_confirm")
