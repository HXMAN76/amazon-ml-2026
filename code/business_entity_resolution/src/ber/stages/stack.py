"""stack: second-stage model with consensus features built from the first-stage probabilities p1.

The pair model scores every (S1, record) pair on its own. Two kinds of evidence only exist once all p1 are known:
  * S1 level: how many of the S1's other candidates are already confident (per source: S2 and S3 records are
    limited to about 5 and 6 matches per S1), the rank of this record inside the S1's list, the gap to the best;
  * record level: how strongly other S1 entities claim the same record, and the margin to the best competitor.
A gradient-boosted model on p1, these consensus features and about twenty carried-over pair features re-scores the
pairs. Train p1 exists for every train S1 (out-of-fold for the sample, unbiased for the rest), so consensus features
are computed at the same density as at test time.

Usage (base = the first-stage model name, e.g. v2; name = the stacked model, e.g. s1):
  python -m ber.stages.stack build   --split train|test   -> WORK/stack/{split}/chunk_*.parquet
  python -m ber.stages.stack tfidf   --split train|test   -> adds tf_name_cos / tf_addr_cos to those chunks (own pass: memory)
  python -m ber.stages.stack train   --name s1              -> WORK/models/s1/{xgb.json,config.json,holdout.json}
  python -m ber.stages.stack predict --name s1              -> WORK/output/s1/{matching_results.tsv,candidate_pairs.tsv}
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.stages.block import PID_BASE
from ber.stages.predict import emit
from ber.split import holdout_q
from ber.stages.train_gpu import pick_device
from ber.tracking import log_stage

# pair features carried over from the first stage (all exist in features/{split}/part_*.parquet)
ORIG = ["score", "ns", "rank_q", "margin_p", "house_eq", "house_lev", "addr_b_empty", "core_eq", "name_tset", "addr_tset",
        "name_jw", "num_common_frac", "digits_ratio", "legal_conflict", "pin_conflict", "pin_match", "nl_name_b",
        "same_ctry", "log_cnt_s1_a", "log_cnt_s1_b", "log_cnt_pool_b", "name_cov_a", "name_cov_b", "rom_tset", "alias_tset"]
# further first-stage columns carried when present (channel columns of the dense retrievers, coverage, skeleton and length features)
EXTRA = ["dall_cos", "dall_rank", "emb_cos", "emb_rank", "addr_cov_a", "addr_cov_b", "skel_ratio", "addr_skel_ratio", "rom_partial",
         "rom_jw", "nospace_partial", "nospace_jw", "len_core_a", "len_core_b", "len_addr_b", "digits_lev", "ntok_addr_a", "ntok_addr_b",
         "legal_eq", "num_common"]
HI = 0.5
STACK_DIR = "stack"  # WORK sub-folder of the chunk files; `--tag X` uses stackX so that variants can be built side by side


def load_p1(split: str, base: str) -> pl.DataFrame:
    """Probabilities of every candidate pair of a split: (q, pid, p[, label]). `base` is a first-stage model, or a stacked model for
    a second consensus round (then `p` is the stacked probability, out-of-fold for training S1, and `p_first` is the first-stage one)."""
    P = config.paths()
    mdl0 = P["work"] / "models" / base
    if (mdl0 / "oof_tune.parquet").exists():  # stacked base: iterate the consensus on its probabilities
        first = json.loads((mdl0 / "config.json").read_text())["base"]
        if split == "test":
            d = pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet")
        else:
            d = pl.concat([pl.read_parquet(mdl0 / "oof_tune.parquet").select("q", "pid", "p", "label"),
                           pl.read_parquet(mdl0 / "holdout_pred.parquet").select("q", "pid", "p", "label")])
        d = d.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("p").cast(pl.Float32))
        pf = load_p1(split, first).select("q", "pid", pl.col("p").alias("p_first"))
        return d.join(pf, on=["q", "pid"], how="left")
    if split == "test":
        d = pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet")
    else:
        mdl = P["work"] / "models" / base
        oof = pl.read_parquet(mdl / "oof.parquet").select("q", "pid", "p", "label")
        rest = [pl.read_parquet(f).select("q", "pid", "p", "label") for f in sorted((mdl / "p1_rest").glob("part_*.parquet"))]
        d = pl.concat([oof, *rest])
    return d.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("p").cast(pl.Float32))


def shortlist(d: pl.DataFrame, prm: dict) -> pl.DataFrame:
    """Candidate shortlist by first-stage probability: per S1 the best `shortlist_k` pairs with p1 >= `shortlist_pmin`, and always the
    best pair (so no S1 has an empty candidate list). Used for training, scoring and the reported candidate_pairs.tsv alike."""
    k, pmin = prm.get("shortlist_k"), prm.get("shortlist_pmin", 0.0)
    if not k:
        return d
    r = pl.col("p").rank("ordinal", descending=True).over("q")
    return d.filter((r == 1) | ((r <= k) & (pl.col("p") >= pmin)))


def pid_features(d: pl.DataFrame) -> pl.DataFrame:
    """Record-level consensus: competition of S1 entities for the same S2/S3 record (needs all S1)."""
    g = d.group_by("pid").agg(
        pl.col("p").sum().alias("sum_p_pid"), pl.col("p").sort(descending=True).head(2).alias("_t"),
        (pl.col("p") > 0.3).sum().cast(pl.Float32).alias("n_claim03"), (pl.col("p") > HI).sum().cast(pl.Float32).alias("n_claim05"),
    ).with_columns(pl.col("_t").list.get(0).alias("_t0"), pl.col("_t").list.get(1, null_on_oob=True).alias("_t1")).drop("_t")
    return (d.join(g, on="pid", how="left")
             .with_columns(pl.when(pl.col("p") >= pl.col("_t0")).then(pl.col("_t1").fill_null(0.0)).otherwise(pl.col("_t0")).alias("max_other_pid"),
                           pl.col("p").rank("ordinal", descending=True).over("pid").cast(pl.Float32).alias("rank_p1_pid"))
             .with_columns((pl.col("p") - pl.col("max_other_pid")).alias("margin_pid")).drop("_t0", "_t1"))


def q_features(d: pl.DataFrame) -> pl.DataFrame:
    """S1-level consensus from the S1's other candidates (per source, with the capacity limits of the data)."""
    src = (pl.col("pid") // PID_BASE).cast(pl.Int8)
    hi = (pl.col("p") > HI).cast(pl.Float32)
    d = d.with_columns(src.alias("_src"), hi.alias("_hi"))
    d = d.with_columns(
        pl.col("p").sum().over("q").alias("sum_p_q"),
        pl.col("_hi").sum().over("q").alias("n_hi_q"),
        (pl.col("_hi") * (pl.col("_src") == 2)).sum().over("q").alias("n_hi_s2"),
        (pl.col("_hi") * (pl.col("_src") == 3)).sum().over("q").alias("n_hi_s3"),
        pl.col("p").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("rank_p1_q"),
        pl.col("p").rank("ordinal", descending=True).over(["q", "_src"]).cast(pl.Float32).alias("rank_p1_q_src"),
        pl.col("p").max().over("q").alias("top_p_q"),
        (pl.col("p") * pl.col("_hi")).sum().over("q").alias("_sum_hi"),
    )
    d = d.with_columns(
        (pl.col("top_p_q") - pl.col("p")).alias("gap_top_q"),
        (pl.col("_sum_hi") - pl.col("p") * pl.col("_hi")).alias("sum_hi_excl"),
        (pl.col("n_hi_q") - pl.col("_hi")).alias("n_hi_excl"),
        (pl.col("sum_p_q") - pl.col("p")).alias("sum_p_excl"),
        pl.when(pl.col("_src") == 2).then(pl.col("n_hi_s2") - pl.col("_hi")).otherwise(pl.col("n_hi_s3") - pl.col("_hi")).alias("n_hi_same_src_excl"),
        pl.col("_src").cast(pl.Float32).alias("src"),
    ).with_columns((pl.col("sum_hi_excl") / pl.col("n_hi_excl").clip(lower_bound=1.0)).alias("mean_hi_excl"))
    return d.drop("_src", "_hi", "_sum_hi")


def _addr_arrays(split: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalised addresses of S1 (by rid) and of the pool (S2 then S3, rows in file order) plus the S2 row count."""
    P = config.paths()
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "addr"]).sort("rid")["addr"].to_numpy()
    s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", "addr"]).sort("rid")["addr"].to_numpy()
    s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", "addr"]).sort("rid")["addr"].to_numpy()
    return s1, np.concatenate([s2, s3]), len(s2)


def _name_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Core names of S1 (by rid) and of the pool (S2 then S3)."""
    P = config.paths()
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1"]).sort("rid")["core1"].to_numpy()
    s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", "core1"]).sort("rid")["core1"].to_numpy()
    s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", "core1"]).sort("rid")["core1"].to_numpy()
    return s1, np.concatenate([s2, s3])


def consensus_text_features(rows: pl.DataFrame, pool_name: np.ndarray, pool_addr: np.ndarray, n2: int) -> pl.DataFrame:
    """Similarity of each candidate to the S1's best OTHER record (highest p1): a look-alike distractor tends to be the odd
    one out among the S1's confident records, while true records resemble each other."""
    from rapidfuzz import fuzz, process

    d = rows.select("q", "pid", "p").with_columns(pl.col("p").rank("ordinal", descending=True).over("q").alias("_r"))
    t1 = pl.col("pid").filter(pl.col("_r") == 1).first().over("q")
    t2 = pl.col("pid").filter(pl.col("_r") == 2).first().over("q")
    p1 = pl.col("p").filter(pl.col("_r") == 1).first().over("q")
    p2 = pl.col("p").filter(pl.col("_r") == 2).first().over("q")
    d = d.with_columns(pl.when(pl.col("_r") == 1).then(t2).otherwise(t1).alias("bo_pid"),
                       pl.when(pl.col("_r") == 1).then(p2).otherwise(p1).alias("bo_p"))
    pid = d["pid"].to_numpy()
    bo = d["bo_pid"].fill_null(0).to_numpy()
    idx = lambda x: np.where(x < 3 * PID_BASE, x - 2 * PID_BASE, n2 + x - 3 * PID_BASE)  # noqa: E731
    ok = (d["bo_p"].fill_null(0.0).to_numpy() > 0.3) & (bo > 0)
    ci, bi = idx(pid), idx(np.where(ok, bo, pid))
    cn, bn = pool_name[ci].tolist(), pool_name[bi].tolist()
    ca, ba = pool_addr[ci].tolist(), pool_addr[bi].tolist()
    f = {}
    for nm, (x, y, sc) in {"cons_name_ratio": (cn, bn, fuzz.ratio), "cons_name_tset": (cn, bn, fuzz.token_set_ratio),
                           "cons_addr_ratio": (ca, ba, fuzz.ratio), "cons_addr_tset": (ca, ba, fuzz.token_set_ratio)}.items():
        v = process.cpdist(x, y, scorer=sc, dtype=np.float32, workers=-1)
        f[nm] = np.where(ok, v, np.nan).astype(np.float32)
    f["bo_p"] = np.where(ok, d["bo_p"].fill_null(0.0).to_numpy(), np.nan).astype(np.float32)
    return pl.DataFrame({"q": d["q"].to_numpy(), "pid": pid, **f})


def decoy_features(rows: pl.DataFrame, s1_name: np.ndarray, pool_name: np.ndarray, n2: int) -> pl.DataFrame:
    """Edit-type features of the core names. A true variant differs mostly by dropped or added characters and words, a look-alike
    distractor keeps the length and swaps letters ("jarlent" and "jarleix"): Levenshtein, Indel, the substitution estimate
    (Indel minus Levenshtein), Hamming distance for equal lengths, length difference, first-letter match, word symmetric difference."""
    from rapidfuzz import process
    from rapidfuzz.distance import Hamming, Indel, Levenshtein

    q, pid = rows["q"].to_numpy(), rows["pid"].to_numpy()
    pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
    a, b = [x or "" for x in s1_name[q].tolist()], [x or "" for x in pool_name[pidx].tolist()]
    lev = process.cpdist(a, b, scorer=Levenshtein.distance, dtype=np.float32, workers=-1)
    ind = process.cpdist(a, b, scorer=Indel.distance, dtype=np.float32, workers=-1)
    la, lb = np.fromiter((len(x) for x in a), dtype=np.float32, count=len(a)), np.fromiter((len(x) for x in b), dtype=np.float32, count=len(b))
    same = la == lb
    ham = process.cpdist(a, b, scorer=Hamming.distance, dtype=np.float32, workers=-1)
    first = np.fromiter((bool(x) and bool(y) and x[0] == y[0] for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    sym = np.fromiter((len(set(x.split()) ^ set(y.split())) for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    return pl.DataFrame({"q": q, "pid": pid, "dc_lev": lev, "dc_indel": ind, "dc_sub": ind - lev, "dc_lendiff": np.abs(la - lb),
                         "dc_same_len": same.astype(np.float32), "dc_ham": np.where(same, ham, np.nan).astype(np.float32),
                         "dc_first": first, "dc_wordsym": sym})


def addr_group_counts(addr: np.ndarray) -> np.ndarray:
    """How many records of one file share each record's exact normalised address (0 for an empty address)."""
    c = pl.DataFrame({"a": addr}).with_columns(pl.col("a").len().over("a").alias("n"), (pl.col("a").str.len_chars() == 0).alias("e"))
    return c.select(pl.when(pl.col("e")).then(0).otherwise(pl.col("n")).cast(pl.Float32)).to_series().to_numpy()


def addr_group_features(rows: pl.DataFrame, s1_addr: np.ndarray, pool_addr: np.ndarray, n2: int, s1_n: np.ndarray, pool_n: np.ndarray) -> pl.DataFrame:
    """Address multiplicity. In France many businesses share one building address (9.9% of pool records share their address with at
    least four others, against 1% in the US and India), so an address match is weaker evidence than in the training countries."""
    q = rows["q"].to_numpy()
    pid = rows["pid"].to_numpy()
    pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
    a, b = s1_n[q], pool_n[pidx]
    same = (s1_addr[q] == pool_addr[pidx]) & (a > 0)
    return pl.DataFrame({"q": q, "pid": pid, "s1_addr_n": np.log1p(a), "pool_addr_n": np.log1p(b), "addr_same_exact": same.astype(np.float32),
                         "addr_n_prod": np.log1p(a) + np.log1p(b)}).with_columns(pl.col("s1_addr_n", "pool_addr_n", "addr_n_prod").cast(pl.Float32))


def digit_features(rows: pl.DataFrame, s1_addr: np.ndarray, pool_addr: np.ndarray, n2: int) -> pl.DataFrame:
    """House-number relations and digit consensus. Look-alike distractors often differ from the S1 by a house number that
    lost or gained a trailing digit, while true records carry other kinds of noise; agreement with the S1's other
    confident records tells which number the S1 really has."""
    q = rows["q"].to_numpy()
    pid = rows["pid"].to_numpy()
    pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
    d = rows.select("q", "pid", "p").with_columns(pl.Series("aa", s1_addr[q]), pl.Series("ab", pool_addr[pidx]))
    d = d.with_columns(
        pl.col("aa").str.extract(r"(\d+)", 1).alias("ha"), pl.col("ab").str.extract(r"(\d+)", 1).alias("hb"),
        pl.col("aa").str.replace_all(r"\D", "").alias("da"), pl.col("ab").str.replace_all(r"\D", "").alias("db"))
    hasa, hasb = pl.col("ha").is_not_null(), pl.col("hb").is_not_null()
    both = hasa & hasb
    d = d.with_columns(
        hasa.cast(pl.Float32).alias("has_h_a"), hasb.cast(pl.Float32).alias("has_h_b"),
        (both & (pl.col("ha") == pl.col("hb"))).cast(pl.Float32).alias("h_eq"),
        (both & pl.col("hb").str.starts_with(pl.col("ha")) & (pl.col("hb").str.len_chars() > pl.col("ha").str.len_chars())).cast(pl.Float32).alias("h_a_prefix_of_b"),
        (both & pl.col("ha").str.starts_with(pl.col("hb")) & (pl.col("ha").str.len_chars() > pl.col("hb").str.len_chars())).cast(pl.Float32).alias("h_b_prefix_of_a"),
        (pl.col("hb").str.len_chars() - pl.col("ha").str.len_chars()).cast(pl.Float32).alias("h_len_diff"),
        (both & pl.col("db").str.starts_with(pl.col("da")) & (pl.col("db") != pl.col("da")) & (pl.col("da") != "")).cast(pl.Float32).alias("d_a_prefix_of_b"),
        (both & pl.col("da").str.starts_with(pl.col("db")) & (pl.col("da") != pl.col("db")) & (pl.col("db") != "")).cast(pl.Float32).alias("d_b_prefix_of_a"),
        (both & (pl.col("da") == pl.col("db"))).cast(pl.Float32).alias("d_eq"),
        (pl.col("hb").cast(pl.Float64, strict=False) - pl.col("ha").cast(pl.Float64, strict=False)).abs().add(1).log().cast(pl.Float32).alias("h_logdiff"),
        (both & pl.col("ab").str.contains(pl.col("ha"), literal=True)).cast(pl.Float32).alias("h_a_in_addr_b"),
    )
    # agreement with the S1's other confident records: which house number does the S1 really have?
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("q").alias("_r"))
    h0 = pl.col("hb").filter(pl.col("_r") == 1).first().over("q")
    h1 = pl.col("hb").filter(pl.col("_r") == 2).first().over("q")
    d = d.with_columns(pl.when(pl.col("_r") == 1).then(h1).otherwise(h0).alias("h_best_other"))
    hbo = pl.col("h_best_other")
    hi = (pl.col("p") > HI)
    d = d.with_columns(
        (hasb & hbo.is_not_null() & (pl.col("hb") == hbo)).cast(pl.Float32).alias("h_eq_best_other"),
        (hasb & hbo.is_not_null() & (pl.col("hb") != hbo) & (hbo.str.starts_with(pl.col("hb")) | pl.col("hb").str.starts_with(hbo))).cast(pl.Float32).alias("h_prefix_best_other"),
        (hasb & hi).cast(pl.Float32).sum().over(["q", "hb"]).alias("_same"))
    d = d.with_columns(
        pl.when(hasb).then(pl.col("_same") - hi.cast(pl.Float32)).otherwise(0.0).alias("n_conf_same_house"),
        (hasb & hi & hasb).cast(pl.Float32).sum().over("q").alias("_conf_h"))
    d = d.with_columns(pl.when(hasb).then(pl.col("_conf_h") - (hasb & hi).cast(pl.Float32) - pl.col("n_conf_same_house")).otherwise(0.0).clip(lower_bound=0.0).alias("n_conf_diff_house"))
    keep = ["has_h_a", "has_h_b", "h_eq", "h_a_prefix_of_b", "h_b_prefix_of_a", "h_len_diff", "d_a_prefix_of_b", "d_b_prefix_of_a", "d_eq",
            "h_logdiff", "h_a_in_addr_b", "h_eq_best_other", "h_prefix_best_other", "n_conf_same_house", "n_conf_diff_house"]
    return d.select(["q", "pid", *keep])


def _hash_docs(texts: list[str], kind: str, workers: int = 4, chunk: int = 400_000):
    """Hashed count vectors (CSR) of a list of strings: character 3-grams for names, word 1-2-grams for addresses."""
    from concurrent.futures import ProcessPoolExecutor

    import scipy.sparse as sp

    parts = [(texts[i: i + chunk], kind) for i in range(0, len(texts), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        mats = list(ex.map(_hash_part, parts))
    return sp.vstack(mats, format="csr")


def _hash_part(args):
    """Worker: vectorise one chunk of strings."""
    from sklearn.feature_extraction.text import HashingVectorizer

    texts, kind = args
    if kind == "name":
        v = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20, alternate_sign=False, norm=None, dtype=np.float32)
    else:
        v = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", n_features=2 ** 20, alternate_sign=False, norm=None, dtype=np.float32)
    return v.transform(texts)


def tfidf_mats(kind: str, s1_texts: list[str], pool_texts: list[str]) -> tuple:
    """TF-IDF (IDF from the pool), L2-normalised in place, for S1 and pool strings of one kind ("name" or "addr")."""
    from sklearn.preprocessing import normalize

    pool = _hash_docs(pool_texts, kind).tocsr()
    df = np.bincount(pool.indices, minlength=pool.shape[1]).astype(np.float32)
    idf = (np.log((pool.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    pool.data *= idf[pool.indices]
    normalize(pool, norm="l2", copy=False)
    s1 = _hash_docs(s1_texts, kind).tocsr()
    s1.data *= idf[s1.indices]
    normalize(s1, norm="l2", copy=False)
    return s1, pool


def tfidf_cos(rows: pl.DataFrame, s1m, poolm, n2: int, step: int = 500_000) -> np.ndarray:
    """TF-IDF cosine of the S1 and pool string for every pair (q, pid) of `rows`."""
    q = rows["q"].to_numpy()
    pid = rows["pid"].to_numpy()
    pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
    v = np.empty(len(q), dtype=np.float32)
    for a in range(0, len(q), step):
        v[a: a + step] = np.asarray(s1m[q[a: a + step]].multiply(poolm[pidx[a: a + step]]).sum(axis=1)).ravel()
    return v


def tfidf_pass(split: str) -> None:
    """Add tf_name_cos and tf_addr_cos to the chunk files of a split, one feature kind at a time (bounded memory)."""
    import gc

    P = config.paths()
    pq = P["parquet"] / split
    files = sorted((P["work"] / STACK_DIR / split).glob("chunk_*.parquet"))
    n2 = pl.read_parquet(pq / "source2.parquet", columns=["rid"]).height
    for kind, col in (("name", "core1"), ("addr", "addr")):
        t0 = time.time()
        get = lambda src: pl.read_parquet(pq / f"source{src}.parquet", columns=["rid", col]).sort("rid")[col].to_list()  # noqa: E731
        pool_texts = get(2) + get(3)
        s1_texts = get(1)
        s1m, poolm = tfidf_mats(kind, s1_texts, pool_texts)
        del pool_texts, s1_texts
        gc.collect()
        print(f"{split} {kind}: matrices built in {time.time() - t0:.0f}s ({poolm.nnz} nonzeros in the pool)", flush=True)
        for f in files:
            d = pl.read_parquet(f)
            d = d.with_columns(pl.Series(f"tf_{kind}_cos", tfidf_cos(d, s1m, poolm, n2)))
            d.write_parquet(f, compression="zstd")
        del s1m, poolm
        gc.collect()
        print(f"{split} {kind}: cosines added to {len(files)} chunk files in {time.time() - t0:.0f}s", flush=True)


def build(split: str, base: str, prm: dict) -> None:
    """Write consensus-feature chunks for a split under WORK/stack/{split}."""
    P = config.paths()
    t0 = time.time()
    d = shortlist(load_p1(split, base), prm)
    print(f"{split}: shortlist keeps {d.height} pairs, {d.height / d['q'].n_unique():.2f} per S1", flush=True)
    dx = None
    if prm.get("xcons") and prm.get("xenc"):  # competition features on the cross-encoder refined probability (xs inside the band, p1 elsewhere)
        xs0 = pl.read_parquet(P["work"] / prm.get("xenc_dir", "xenc") / f"{split}_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
        if split == "train":  # xs of the S1 the cross-encoder was fitted on is optimistic: keep it out of the competitors' refined probabilities
            from ber.stages.xenc import xenc_train_q

            xp0 = dict(config.load()["xenc"])
            if prm.get("xenc_fit_more"):
                xp0["fit_more"] = prm["xenc_fit_more"]
            xs0 = xs0.filter(~pl.col("q").is_in(xenc_train_q(xp0)))
        dx = (d.select("q", "pid", "p").join(xs0, on=["q", "pid"], how="left")
               .with_columns(pl.when(pl.col("xs").is_not_null()).then(pl.col("xs")).otherwise(pl.col("p")).cast(pl.Float32).alias("p")).select("q", "pid", "p"))
        dx = pid_features(dx)
    d = pid_features(d)
    if split == "train":  # keep the locked holdout plus a seeded subsample of the other S1 for training
        hold = holdout_q()
        allq = d.select("q").unique()["q"].to_numpy()
        pool = np.setdiff1d(allq, hold)
        dense_ft = config.load().get("dense")
        if dense_ft:  # S1 the name encoder was fine-tuned on would carry optimistic emb_cos features: keep them out of stage two
            from ber.stages.dense import dense_train_q

            pool = np.setdiff1d(pool, dense_train_q(dense_ft))
            if config.load().get("dense_all"):
                from ber.stages.dense_all import dense_all_train_q

                pool = np.setdiff1d(pool, dense_all_train_q(config.load()["dense_all"]))
        if prm.get("xenc"):  # S1 the cross-encoder was fitted on would carry optimistic xs features
            from ber.stages.xenc import xenc_train_q

            xp = dict(config.load()["xenc"])
            if prm.get("xenc_fit_more"):
                xp["fit_more"] = prm["xenc_fit_more"]
            pool = np.setdiff1d(pool, xenc_train_q(xp))
        rng = np.random.default_rng(prm["seed"])
        sub = rng.choice(pool, size=min(prm["sub_q"], len(pool)), replace=False)
        keep = np.sort(np.concatenate([hold, sub]))
        d = d.join(pl.DataFrame({"q": keep}), on="q", how="semi")
        feat_files = [str(f) for sub_dir in ("train", "train_rest") for f in sorted((P["work"] / "features" / sub_dir).glob("part_*.parquet"))]
    else:
        keep = np.sort(d.select("q").unique()["q"].to_numpy())
        if prm.get("ctry"):  # rebuild one country's S1 only (competition features above still see every claimant)
            cq = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).filter(pl.col("ctry") == prm["ctry"])["rid"].to_numpy()
            keep = np.intersect1d(keep, cq.astype(keep.dtype))
        feat_files = [str(f) for f in sorted((P["work"] / "features" / "test").glob("part_*.parquet"))]
    d = d.sort("q", "pid")
    out = P["work"] / STACK_DIR / split
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    scan = pl.scan_parquet(feat_files)
    s1_addr, pool_addr, n2 = _addr_arrays(split)
    s1_name, pool_name = _name_arrays(split)
    xs_df = pl.read_parquet(P["work"] / prm.get("xenc_dir", "xenc") / f"{split}_xs.parquet") if prm.get("xenc") else None
    if xs_df is not None and prm.get("xenc_dir2"):  # a second cross-encoder as a second feature xs2
        x2 = pl.read_parquet(P["work"] / prm["xenc_dir2"] / f"{split}_xs.parquet").select("q", "pid", pl.col("xs").alias("xs2"))
        xs_df = xs_df.join(x2.with_columns(pl.col("q").cast(xs_df["q"].dtype), pl.col("pid").cast(xs_df["pid"].dtype)), on=["q", "pid"], how="full", coalesce=True)
    if xs_df is not None and prm.get("xenc_dir3"):  # a third cross-encoder (different family) as xs3, scored on the band pairs only
        x3 = pl.read_parquet(P["work"] / prm["xenc_dir3"] / f"{split}_xs.parquet").select("q", "pid", pl.col("xs").alias("xs3"))
        xs_df = xs_df.join(x3.with_columns(pl.col("q").cast(xs_df["q"].dtype), pl.col("pid").cast(xs_df["pid"].dtype)), on=["q", "pid"], how="full", coalesce=True)
    if prm.get("addrmult"):
        s1_addr_n, pool_addr_n = addr_group_counts(s1_addr), addr_group_counts(pool_addr)
    have = set(scan.collect_schema().names())
    carried = [c for c in dict.fromkeys(ORIG + (EXTRA if prm.get("extra_features", False) else [])) if c in have]
    n = 0
    for i, lo_i in enumerate(range(0, len(keep), prm["chunk_q"])):
        qs = keep[lo_i: lo_i + prm["chunk_q"]]
        lo, hi = int(qs[0]), int(qs[-1])
        rows0 = d.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).join(pl.DataFrame({"q": qs}), on="q", how="semi")
        rows = q_features(rows0)
        if dx is not None:
            rx = q_features(dx.join(pl.DataFrame({"q": qs}), on="q", how="semi").filter((pl.col("q") >= lo) & (pl.col("q") <= hi)))
            xc = [c for c in rx.columns if c not in {"q", "pid", "p"}]
            rows = rows.join(rx.select("q", "pid", pl.col("p").alias("px"), *xc).rename({c: c + "_x" for c in xc}), on=["q", "pid"], how="left")
        orig = (scan.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).select(["q", "pid", *carried])
                    .with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).collect())
        rows = rows.join(orig, on=["q", "pid"], how="left").join(digit_features(rows0, s1_addr, pool_addr, n2), on=["q", "pid"], how="left")
        rows = rows.join(consensus_text_features(rows0, pool_name, pool_addr, n2), on=["q", "pid"], how="left")
        if xs_df is not None:
            rows = rows.join(xs_df.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)), on=["q", "pid"], how="left")
        if prm.get("addrmult"):
            rows = rows.join(addr_group_features(rows0, s1_addr, pool_addr, n2, s1_addr_n, pool_addr_n), on=["q", "pid"], how="left")
        if prm.get("decoy", False):
            rows = rows.join(decoy_features(rows0, s1_name, pool_name, n2), on=["q", "pid"], how="left")
        assert rows.height == rows0.height, "feature rows and first-stage rows must match one to one"
        rows = rows.with_columns([pl.col(c).cast(pl.Float32) for c in rows.columns if c not in {"q", "pid", "label"}])
        if "label" in rows.columns:
            rows = rows.with_columns(pl.col("label").cast(pl.Int8))
        rows.write_parquet(out / f"chunk_{i:04d}.parquet", compression="zstd")
        n += rows.height
        print(f"{split} chunk {i}: {rows.height} pairs, {rows.width} columns", flush=True)
    log_stage(f"stack_build_{split}", prm, {"pairs": float(n), "seconds": time.time() - t0})


def _read(split: str) -> pl.DataFrame:
    P = config.paths()
    return pl.concat([pl.read_parquet(f) for f in sorted((P["work"] / STACK_DIR / split).glob("chunk_*.parquet"))])


def _load_arrays(split: str, drop: tuple[str, ...] = (), exact: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Chunk files straight into one preallocated float32 matrix (no polars concat, no second copy)."""
    P = config.paths()
    files = sorted((P["work"] / STACK_DIR / split).glob("chunk_*.parquet"))
    first = pl.read_parquet(files[0])
    feats = [c for c in first.columns if c not in {"q", "pid", "label"} and not c.startswith(drop) and c not in exact]
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    x = np.empty((n, len(feats)), dtype=np.float32)
    qa, pida, y = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int8)
    a = 0
    for f in files:
        d = pl.read_parquet(f)
        m = d.height
        x[a: a + m] = d.select(feats).to_numpy()
        qa[a: a + m], pida[a: a + m] = d["q"].to_numpy(), d["pid"].to_numpy()
        y[a: a + m] = d["label"].to_numpy() if "label" in d.columns else 0
        a += m
    return x, qa, pida, y, feats


def train(name: str, base: str, prm: dict, drop: tuple[str, ...] = ()) -> None:
    """Fit the stacked model on the non-holdout S1, tune the threshold, and score the locked holdout with a paired CI."""
    P = config.paths()
    t0 = time.time()
    hold = pl.DataFrame({"q": holdout_q()})
    x, qa, pida, y, feats = _load_arrays("train", drop, tuple(prm.get("drop_exact", ())))  # `drop`: feature prefixes left out (ablations)
    is_hold = np.isin(qa, hold["q"].to_numpy())
    q = qa.astype(np.uint64)
    fold = ((q * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(prm["folds"])).astype(np.int64)
    device = pick_device(prm["device"])
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "device": device, "tree_method": "hist", "max_depth": prm["max_depth"],
         "eta": prm["eta"], "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "seed": prm["seed"]}
    tr_mask = ~is_hold
    oof = np.zeros(len(y), dtype=np.float32)
    iters = []
    for k in range(prm["folds"]):
        tr, va = tr_mask & (fold != k), tr_mask & (fold == k)
        dtr = xgb.QuantileDMatrix(x[tr], label=y[tr], feature_names=feats)
        m = xgb.train(p, dtr, prm["rounds"],
                      evals=[(xgb.QuantileDMatrix(x[va], label=y[va], feature_names=feats, ref=dtr), "val")], early_stopping_rounds=prm["early_stop"], verbose_eval=False)
        del dtr
        oof[va] = m.predict(xgb.DMatrix(x[va], feature_names=feats), iteration_range=(0, m.best_iteration + 1))
        iters.append(m.best_iteration + 1)
        print(f"fold {k}: {iters[-1]} rounds, aucpr {m.best_score:.4f}", flush=True)
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    tune = pl.DataFrame({"q": qa[tr_mask], "pid": pida[tr_mask], "p": oof[tr_mask], "label": y[tr_mask]})
    nt_tune = tune.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    thr, score_oof, _ = decision.tune_threshold(tune, nt_tune, True)
    print(f"stacked OOF macro F0.5 on the non-holdout subsample: {score_oof:.4f} at threshold {thr:.2f}", flush=True)
    dfinal = xgb.QuantileDMatrix(x[tr_mask], label=y[tr_mask], feature_names=feats)
    model = xgb.train(p, dfinal, int(np.mean(iters) * 1.1) + 1)
    del dfinal

    # locked holdout: baseline (first-stage p1 with its own threshold) versus stacked, on the same S1
    hq, hp, hy = qa[is_hold], pida[is_hold], y[is_hold]
    p2 = model.predict(xgb.DMatrix(x[is_hold], feature_names=feats))
    base_cfg = json.loads((P["work"] / "models" / base / "config.json").read_text())
    nt_h = hold.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    a = pl.DataFrame({"q": hq, "pid": hp, "p": x[is_hold][:, feats.index("p")], "label": hy})
    b = pl.DataFrame({"q": hq, "pid": hp, "p": p2, "label": hy})
    ea = decision.per_entity_f05(decision.assign_exclusive(a).filter(pl.col("p") >= base_cfg["threshold"]), nt_h).sort("q")["f"].to_numpy()
    eb = decision.per_entity_f05(decision.assign_exclusive(b).filter(pl.col("p") >= thr), nt_h).sort("q")["f"].to_numpy()
    delta, lo, hi = decision.paired_bootstrap_delta(ea, eb)
    rep = {"base": base, "base_holdout_f05": float(ea.mean()), "stack_holdout_f05": float(eb.mean()), "delta": delta,
           "delta_ci95": [lo, hi], "ship": bool(lo > 0), "stack_threshold": thr, "oof_subsample_f05": score_oof, "holdout_s1": hold.height}
    out = P["work"] / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    tune.write_parquet(out / "oof_tune.parquet", compression="zstd")   # out-of-fold p on the non-holdout S1 (calibration, tuning)
    b.write_parquet(out / "holdout_pred.parquet", compression="zstd")  # stacked p on the locked holdout
    model.save_model(str(out / "xgb.json"))
    (out / "config.json").write_text(json.dumps({"features": feats, "threshold": thr, "exclusive": True, "device_trained": device, "base": base, "tag": STACK_DIR[len("stack"):]}, indent=2))
    (out / "holdout.json").write_text(json.dumps(rep, indent=2))
    print("HOLDOUT paired comparison:", json.dumps(rep), flush=True)
    imp = model.get_score(importance_type="gain")
    print("top features:", sorted(imp.items(), key=lambda kv: -kv[1])[:12], flush=True)
    log_stage("stack_train", prm, {"holdout_base": float(ea.mean()), "holdout_stack": float(eb.mean()), "delta": delta, "delta_lo": lo,
                                   "seconds": time.time() - t0})


def predict(name: str) -> None:
    """Score the test pairs with the stacked model, apply the decision rule, write and validate the TSV outputs."""
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models" / name
    cfg = json.loads((mdl / "config.json").read_text())
    global STACK_DIR
    STACK_DIR = "stack" + cfg.get("tag", "")
    model = xgb.Booster()
    model.load_model(str(mdl / "xgb.json"))
    if cfg.get("device_trained") == "cuda":
        model.set_param({"device": "cuda"})
    parts = []
    for f in sorted((P["work"] / STACK_DIR / "test").glob("chunk_*.parquet")):
        d = pl.read_parquet(f)
        pp = model.predict(xgb.DMatrix(d.select(cfg["features"]).to_numpy().astype(np.float32), feature_names=cfg["features"]))
        parts.append(d.select("q", "pid").with_columns(pl.Series("p", pp)))
    df = pl.concat(parts)
    (P["work"] / "output" / name).mkdir(parents=True, exist_ok=True)
    df.write_parquet(P["work"] / "output" / name / "pair_p.parquet", compression="zstd")
    emit(name, P, df, cfg, t0)


def main(argv: list[str] | None = None) -> None:
    """CLI: build | train | predict (see module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "tfidf", "train", "predict"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--name", default="s1")
    ap.add_argument("--base", default=None)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to leave out of training (ablation)")
    ap.add_argument("--drop-exact", default="", help="train: comma-separated exact feature names left out of training")
    ap.add_argument("--tag", default="", help="separate chunk folder stack<tag> (parallel variants)")
    ap.add_argument("--decoy", action="store_true", help="build: add the edit-type (decoy) name features")
    ap.add_argument("--extra", action="store_true", help="build: carry more first-stage feature columns")
    ap.add_argument("--set", default="", help="comma-separated stack parameter overrides, e.g. max_depth=9,eta=0.05,rounds=1500")
    ap.add_argument("--xenc", action="store_true", help="build: add the cross-encoder score xs (needs stages/xenc.py output)")
    ap.add_argument("--xenc-dir2", default="", help="build: WORK sub-folder of a second cross-encoder (feature xs2)")
    ap.add_argument("--addrmult", action="store_true", help="build: address multiplicity features (how many records share the address)")
    ap.add_argument("--xenc-dir3", default="", help="build: WORK sub-folder of a third cross-encoder (feature xs3)")
    ap.add_argument("--xcons", action="store_true", help="build: consensus features on the cross-encoder refined probability (needs --xenc)")
    ap.add_argument("--xenc-fit-more", type=int, default=0, help="build: the cross-encoder was fitted on this many more S1 (exclude them)")
    ap.add_argument("--xenc-dir", default="xenc", help="build: WORK sub-folder with the cross-encoder scores")
    ap.add_argument("--ctry", default="", help="build, test split: write the chunks of this country's S1 only (e.g. france)")
    ap.add_argument("--sub-q", type=int, default=None, help="build: number of non-holdout S1 for stage-two training (default: params stack.sub_q)")
    a = ap.parse_args(argv)
    global STACK_DIR
    STACK_DIR = "stack" + a.tag
    prm = dict(config.load()["stack"])
    prm["decoy"] = prm.get("decoy", False) or a.decoy
    prm["extra_features"] = prm.get("extra_features", False) or a.extra
    prm["xenc"] = prm.get("xenc", False) or a.xenc
    prm["xenc_dir"] = a.xenc_dir
    prm["xcons"] = prm.get("xcons", False) or a.xcons
    prm["xenc_dir2"] = a.xenc_dir2
    prm["xenc_dir3"] = a.xenc_dir3
    prm["addrmult"] = a.addrmult
    prm["xenc_fit_more"] = a.xenc_fit_more
    prm["ctry"] = a.ctry
    if a.sub_q:
        prm["sub_q"] = a.sub_q
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        prm[k] = type(prm[k])(v) if k in prm else float(v)
    base = a.base or prm["base"]
    if a.cmd == "build":
        build(a.split, base, prm)
    elif a.cmd == "tfidf":
        tfidf_pass(a.split)
    elif a.cmd == "train":
        prm["drop_exact"] = tuple(x for x in a.drop_exact.split(",") if x)
        train(a.name, base, prm, tuple(x for x in a.drop.split(",") if x))
    else:
        predict(a.name)


if __name__ == "__main__":
    main()
