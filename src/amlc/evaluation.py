"""Metrics used by past Amazon ML Challenges plus common ones. Pick with get_metric(name)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (2025 price challenge). 0/0 counts as 0 error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true)
    return float(100 * np.mean(np.where(denom == 0, 0.0, diff / np.where(denom == 0, 1, denom))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def f1_macro(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def extraction_f1(y_true, y_pred) -> float:
    """2024 entity-extraction F1: empty prediction vs non-empty truth is a false negative,
    non-empty prediction that mismatches (or truth empty) is a false positive."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t = "" if t is None else str(t).strip()
        p = "" if p is None else str(p).strip()
        if p and t and p == t:
            tp += 1
        elif p and (not t or p != t):
            fp += 1
        elif not p and t:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# name -> (fn, greater_is_better)
METRICS: dict[str, tuple[Callable, bool]] = {
    "smape": (smape, False),
    "rmse": (rmse, False),
    "mae": (mae, False),
    "accuracy": (accuracy, True),
    "f1_macro": (f1_macro, True),
    "extraction_f1": (extraction_f1, True),
}


def get_metric(name: str) -> tuple[Callable, bool]:
    return METRICS[name]
