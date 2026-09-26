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


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference b - a of per-entity scores on the same S1 and its 95% paired bootstrap interval."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return bootstrap_ci(d, n_boot=n_boot, seed=seed)


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


def tune_threshold_by_group(df: pl.DataFrame, n_true: pl.DataFrame, groups: pl.DataFrame, exclusive: bool = True,
                            min_q: int = 2000, grid: np.ndarray | None = None) -> dict[str, float]:
    """One threshold per group of S1 (e.g. country). groups: (q, g). Exclusive assignment is decided on all pairs first
    (competition is global); each group then gets its own cut. Groups with fewer than `min_q` S1 get no entry, so they
    fall back to the global threshold when applied (France has no training S1)."""
    base = assign_exclusive(df) if exclusive else df
    out: dict[str, float] = {}
    for (g,), sub_q in groups.group_by("g"):
        if sub_q.height < min_q:
            continue
        nt = n_true.join(sub_q.select("q"), on="q", how="semi")
        thr, _, _ = tune_threshold(base.join(sub_q.select("q"), on="q", how="semi"), nt, exclusive=False, grid=grid)
        out[str(g)] = thr
    return out


def select_by_threshold(df: pl.DataFrame, default: float, by_group: dict[str, float] | None = None,
                        groups: pl.DataFrame | None = None) -> pl.DataFrame:
    """Keep pairs with p >= the threshold of their S1's group (default for groups without their own threshold)."""
    if not by_group or groups is None:
        return df.filter(pl.col("p") >= default)
    tab = pl.DataFrame({"g": list(by_group.keys()), "_thr": list(by_group.values())}, schema={"g": pl.Utf8, "_thr": pl.Float64})
    g = groups.with_columns(pl.col("q").cast(df.schema["q"]), pl.col("g").cast(pl.Utf8)).join(tab, on="g", how="left")
    d = df.join(g.select("q", "_thr"), on="q", how="left").with_columns(pl.col("_thr").fill_null(default))
    return d.filter(pl.col("p") >= pl.col("_thr")).drop("_thr")


def cap_per_source(sel: pl.DataFrame, caps: dict[int, int], pid_base: int = 10_000_000) -> pl.DataFrame:
    """Keep at most caps[src] selected records per S1 and source (highest p first): the training data never has more than
    5 S2 and 6 S3 matches for one S1, so extra selections are errors."""
    if not caps:
        return sel
    src = (pl.col("pid") // pid_base).cast(pl.Int64)
    lim = pl.lit(None, dtype=pl.Int64)
    for s, c in caps.items():
        lim = pl.when(src == int(s)).then(pl.lit(int(c), dtype=pl.Int64)).otherwise(lim)
    d = sel.with_columns(src.alias("_src"), lim.alias("_cap"))
    d = d.sort(["q", "_src", "p", "pid"], descending=[False, False, True, False]).with_columns(
        pl.int_range(pl.len()).over(["q", "_src"]).alias("_r"))
    return d.filter(pl.col("_cap").is_null() | (pl.col("_r") < pl.col("_cap"))).drop("_src", "_cap", "_r")
