"""Candidate generation: top-K TF-IDF nearest neighbours per S1 record, within country.

Two views (core-name char n-grams, name+address char n-grams) are unioned so a record that
matches by name only, or by address only-ish, still gets proposed. Recall of this stage is the
ceiling for the whole pipeline; it is reported on train by `ber.run train`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _topk_rows(sim: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim.shape[1])
    if k == sim.shape[1]:
        return np.tile(np.arange(k), (sim.shape[0], 1))
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def _knn(xa: sparse.csr_matrix, xb: sparse.csr_matrix, k: int, budget: float = 3e7) -> tuple[np.ndarray, np.ndarray]:
    """For each row of xa the top-k rows of xb by dot product. Returns (row_idx, col_idx) pairs."""
    n, m = xa.shape[0], xb.shape[0]
    chunk = max(16, int(budget // max(m, 1)))
    rows, cols = [], []
    xbt = xb.T.tocsc()
    for s in range(0, n, chunk):
        sim = (xa[s : s + chunk] @ xbt).toarray()
        top = _topk_rows(sim, k)
        rows.append(np.repeat(np.arange(s, s + sim.shape[0]), top.shape[1]))
        cols.append(top.ravel())
    return np.concatenate(rows), np.concatenate(cols)


def generate(
    s1: pd.DataFrame,
    oth: pd.DataFrame,
    e1: dict,
    eo: dict,
    k_name: int = 25,
    k_both: int = 25,
    country_block: bool = True,
) -> pd.DataFrame:
    """Returns unique candidate pairs (i = row in s1, j = row in oth)."""
    c1, co = s1["ctry"].to_numpy(), oth["ctry"].to_numpy()
    groups = pd.Series(np.arange(len(s1))).groupby(c1).apply(lambda x: x.to_numpy()) if country_block else {"": np.arange(len(s1))}
    parts = []
    for ctry, rows in groups.items():
        cols = np.where(co == ctry)[0] if country_block and ctry else np.arange(len(oth))
        if len(cols) == 0:  # country never seen on the other side: do not lose the entity, look everywhere
            cols = np.arange(len(oth))
        for key, k in (("core_char", k_name), ("both_char", k_both)):
            r, c = _knn(e1[key][rows], eo[key][cols], k)
            parts.append(pd.DataFrame({"i": rows[r], "j": cols[c]}))
    return pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(["i", "j"]).reset_index(drop=True)


def recall(pairs: pd.DataFrame, s1: pd.DataFrame, oth: pd.DataFrame, truth: dict[str, set[str]]) -> dict[str, float]:
    """Pair-level recall ceiling and reduction ratio of the candidate set."""
    cand = set(zip(s1["entity_id"].to_numpy()[pairs["i"]], oth["entity_id"].to_numpy()[pairs["j"]]))
    total = hit = 0
    for a, ms in truth.items():
        for m in ms:
            total += 1
            hit += (a, m) in cand
    return {
        "candidate_recall": hit / max(total, 1),
        "n_true_pairs": total,
        "n_candidates": len(pairs),
        "cand_per_s1": len(pairs) / max(len(s1), 1),
        "reduction_ratio": 1 - len(pairs) / max(len(s1) * len(oth), 1),
    }
