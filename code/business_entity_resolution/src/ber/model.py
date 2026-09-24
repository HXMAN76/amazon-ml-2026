"""LightGBM pair classifier with grouped (by S1 entity) cross-validation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ber.features import feature_cols

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1,
)


def label_pairs(pairs: pd.DataFrame, s1: pd.DataFrame, oth: pd.DataFrame, truth: dict[str, set[str]]) -> np.ndarray:
    a, b = s1["entity_id"].to_numpy()[pairs["i"]], oth["entity_id"].to_numpy()[pairs["j"]]
    return np.fromiter((y in truth.get(x, ()) for x, y in zip(a, b)), dtype=np.int8, count=len(a))


def cv_oof(feats: pd.DataFrame, y: np.ndarray, folds: int = 5, rounds: int = 800, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold probabilities. Folds are grouped by S1 row so an entity never leaks across folds."""
    cols = feature_cols(feats)
    oof = np.zeros(len(feats))
    best: list[int] = []
    for tr, va in GroupKFold(n_splits=folds).split(feats, y, groups=feats["i"]):
        dtr = lgb.Dataset(feats.iloc[tr][cols], y[tr])
        dva = lgb.Dataset(feats.iloc[va][cols], y[va])
        m = lgb.train({**PARAMS, "seed": seed}, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(feats.iloc[va][cols], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
    return oof, best


def fit_full(feats: pd.DataFrame, y: np.ndarray, rounds: int, seed: int = 0) -> lgb.Booster:
    cols = feature_cols(feats)
    return lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(feats[cols], y), rounds)


def predict(model: lgb.Booster, feats: pd.DataFrame) -> np.ndarray:
    return model.predict(feats[feature_cols(feats)])
