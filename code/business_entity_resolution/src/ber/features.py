"""Vectorisers and pairwise features for candidate (S1, S2/S3) pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ber import normalize as N


class Encoder:
    """TF-IDF spaces fitted on all records of one split (unsupervised, no labels)."""

    def __init__(self) -> None:
        self.core_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.both_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, dtype=np.float32)
        self.name_word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
        self.addr_word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)

    def fit(self, *dfs: pd.DataFrame) -> "Encoder":
        allr = pd.concat(dfs, ignore_index=True)
        self.core_char.fit(allr["core"])
        self.both_char.fit(allr["both"])
        self.name_word.fit(allr["core"])
        self.addr_word.fit(allr["addr"])
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, sparse.csr_matrix]:
        return {
            "core_char": self.core_char.transform(df["core"]).tocsr(),
            "both_char": self.both_char.transform(df["both"]).tocsr(),
            "name_word": self.name_word.transform(df["core"]).tocsr(),
            "addr_word": self.addr_word.transform(df["addr"]).tocsr(),
        }


def rowdot(a: sparse.csr_matrix, b: sparse.csr_matrix, ii: np.ndarray, jj: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Cosine (rows are L2-normalised) for aligned row pairs a[ii[k]] . b[jj[k]]."""
    out = np.empty(len(ii), dtype=np.float32)
    for s in range(0, len(ii), chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.asarray(a[ii[sl]].multiply(b[jj[sl]]).sum(axis=1)).ravel()
    return out


def _jaccard(a: list[set], b: list[set]) -> np.ndarray:
    return np.fromiter(
        (len(x & y) / len(x | y) if (x or y) else 0.0 for x, y in zip(a, b)), dtype=np.float32, count=len(a)
    )


def _cp(qs: list[str], cs: list[str], scorer) -> np.ndarray:
    if not qs:
        return np.zeros(0, dtype=np.float32)
    return process.cpdist(qs, cs, scorer=scorer, dtype=np.float32, workers=-1)


def _rank_within(df: pd.DataFrame, key: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby(key)[col]
    rank = g.rank(method="min", ascending=False).to_numpy(dtype=np.float32)
    gap = (g.transform("max") - df[col]).to_numpy(dtype=np.float32)
    return rank, gap


def pair_features(
    s1: pd.DataFrame, oth: pd.DataFrame, pairs: pd.DataFrame, e1: dict, eo: dict
) -> pd.DataFrame:
    """pairs has columns i (row in s1) and j (row in oth). Returns one feature row per pair."""
    ii, jj = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    a, b = s1.iloc[ii], oth.iloc[jj]
    f: dict[str, np.ndarray] = {}

    cn1, cn2 = a["core"].tolist(), b["core"].tolist()
    n1, n2 = a["name"].tolist(), b["name"].tolist()
    ad1, ad2 = a["addr"].tolist(), b["addr"].tolist()

    for nm, sc in (
        ("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio), ("tsort", fuzz.token_sort_ratio),
        ("tset", fuzz.token_set_ratio),
    ):
        f[f"name_{nm}"] = _cp(cn1, cn2, sc)
        f[f"addr_{nm}"] = _cp(ad1, ad2, sc)
    f["name_jw"] = _cp(cn1, cn2, JaroWinkler.normalized_similarity)
    f["name_lev"] = _cp(cn1, cn2, Levenshtein.normalized_similarity)
    f["fullname_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["addr_lev"] = _cp(ad1, ad2, Levenshtein.normalized_similarity)

    t1, t2 = [set(x.split()) for x in cn1], [set(x.split()) for x in cn2]
    f["name_jac"] = _jaccard(t1, t2)
    f["addr_jac"] = _jaccard([set(x.split()) for x in ad1], [set(x.split()) for x in ad2])
    f["num_jac"] = _jaccard(a["nums"].tolist(), b["nums"].tolist())
    f["num_overlap"] = np.array([len(x & y) for x, y in zip(a["nums"], b["nums"])], dtype=np.float32)
    f["num_both_have"] = np.array([bool(x) and bool(y) for x, y in zip(a["nums"], b["nums"])], dtype=np.float32)
    f["name_eq"] = np.array([x == y for x, y in zip(cn1, cn2)], dtype=np.float32)
    f["first_tok_eq"] = np.array(
        [bool(x) and bool(y) and x.split()[0] == y.split()[0] for x, y in zip(cn1, cn2)], dtype=np.float32
    )
    f["acr_match"] = np.array(
        [
            bool(x) and bool(y) and (N.acronym(x) == y.replace(" ", "") or N.acronym(y) == x.replace(" ", ""))
            for x, y in zip(cn1, cn2)
        ],
        dtype=np.float32,
    )
    pin = lambda s: next((t for t in reversed(s.split()) if t.isdigit() and len(t) >= 5), "")  # noqa: E731
    p1, p2 = [pin(x) for x in ad1], [pin(x) for x in ad2]
    f["pin_match"] = np.array([bool(x) and x == y for x, y in zip(p1, p2)], dtype=np.float32)
    f["pin_conflict"] = np.array([bool(x) and bool(y) and x != y for x, y in zip(p1, p2)], dtype=np.float32)
    f["len_name_1"] = np.array([len(x) for x in cn1], dtype=np.float32)
    f["len_name_2"] = np.array([len(x) for x in cn2], dtype=np.float32)
    f["len_addr_1"] = np.array([len(x) for x in ad1], dtype=np.float32)
    f["len_addr_2"] = np.array([len(x) for x in ad2], dtype=np.float32)
    f["ntok_name_diff"] = np.abs(np.array([len(x) for x in t1]) - np.array([len(x) for x in t2])).astype(np.float32)
    f["same_country"] = (a["ctry"].to_numpy() == b["ctry"].to_numpy()).astype(np.float32)
    f["is_s3"] = b["entity_id"].str.startswith("S3").to_numpy().astype(np.float32)

    for k in ("core_char", "both_char", "name_word", "addr_word"):
        f[f"cos_{k}"] = rowdot(e1[k], eo[k], ii, jj)

    df = pd.DataFrame(f)
    df["i"], df["j"] = ii, jj
    # competition features: how good is this pair relative to the other options of the same S1 / same candidate
    for col in ("cos_both_char", "cos_core_char", "name_tset"):
        r, g = _rank_within(df, "i", col)
        df[f"{col}_rank_i"], df[f"{col}_gap_i"] = r, g
        r, g = _rank_within(df, "j", col)
        df[f"{col}_rank_j"], df[f"{col}_gap_j"] = r, g
    df["n_cand_i"] = df.groupby("i")["i"].transform("size").astype(np.float32)
    df["n_prop_j"] = df.groupby("j")["j"].transform("size").astype(np.float32)
    return df


FEATURE_EXCLUDE = {"i", "j", "label"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
