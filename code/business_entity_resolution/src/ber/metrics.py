"""Competition metric: macro-averaged F_0.5 over Source 1 entities (singletons included)."""

from __future__ import annotations

BETA2 = 0.25


def f05_entity(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    ids = list(truth)
    return sum(f05_entity(pred.get(i, set()), truth[i]) for i in ids) / max(len(ids), 1)


def breakdown(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> dict[str, float]:
    """Score split by singleton / non-singleton S1 entities, to see where points are lost."""
    single = [i for i in truth if not truth[i]]
    multi = [i for i in truth if truth[i]]

    def m(ids: list[str]) -> float:
        return sum(f05_entity(pred.get(i, set()), truth[i]) for i in ids) / max(len(ids), 1)

    return {
        "macro_f05": macro_f05(pred, truth),
        "singleton_f05": m(single),
        "matched_f05": m(multi),
        "n_singleton": len(single),
        "n_matched": len(multi),
    }
