"""Turn pair probabilities into final match sets, tuned for macro F_0.5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ber.metrics import macro_f05


def select(pairs: pd.DataFrame, p: np.ndarray, s1_ids: np.ndarray, oth_ids: np.ndarray,
           thr: float, one_to_one: bool) -> dict[str, set[str]]:
    """Keep pairs with p >= thr. With one_to_one, an S2/S3 record is kept only under its best S1."""
    d = pd.DataFrame({"i": pairs["i"].to_numpy(), "j": pairs["j"].to_numpy(), "p": p})
    if one_to_one:
        d = d.loc[d.groupby("j")["p"].idxmax()]
    d = d[d["p"] >= thr]
    out: dict[str, set[str]] = {e: set() for e in s1_ids}
    for i, j in zip(d["i"].to_numpy(), d["j"].to_numpy()):
        out[s1_ids[i]].add(oth_ids[j])
    return out


def tune(pairs: pd.DataFrame, p: np.ndarray, s1_ids: np.ndarray, oth_ids: np.ndarray,
         truth: dict[str, set[str]], grid: np.ndarray | None = None) -> dict:
    """Grid-search the threshold (and one-to-one on/off) on out-of-fold predictions."""
    grid = np.arange(0.2, 0.96, 0.02) if grid is None else grid
    best = {"score": -1.0}
    table = []
    for o2o in (False, True):
        for t in grid:
            s = macro_f05(select(pairs, p, s1_ids, oth_ids, float(t), o2o), truth)
            table.append((o2o, float(t), s))
            if s > best["score"]:
                best = {"score": s, "thr": float(t), "one_to_one": o2o}
    best["table"] = table
    return best
