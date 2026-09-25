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
from ber.stages.score_rest import holdout_q
from ber.stages.train_gpu import pick_device
from ber.tracking import log_stage

# pair features carried over from the first stage (all exist in features/{split}/part_*.parquet)
ORIG = ["score", "ns", "rank_q", "margin_p", "house_eq", "house_lev", "addr_b_empty", "core_eq", "name_tset", "addr_tset",
        "name_jw", "num_common_frac", "digits_ratio", "legal_conflict", "pin_conflict", "pin_match", "nl_name_b",
        "same_ctry", "log_cnt_s1_a", "log_cnt_s1_b", "log_cnt_pool_b", "name_cov_a", "name_cov_b", "rom_tset", "alias_tset"]
HI = 0.5


def load_p1(split: str, base: str) -> pl.DataFrame:
    """First-stage probabilities of every candidate pair of a split: (q, pid, p[, label])."""
    P = config.paths()
    if split == "test":
        d = pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet")
    else:
        mdl = P["work"] / "models" / base
        oof = pl.read_parquet(mdl / "oof.parquet").select("q", "pid", "p", "label")
        rest = [pl.read_parquet(f).select("q", "pid", "p", "label") for f in sorted((mdl / "p1_rest").glob("part_*.parquet"))]
        d = pl.concat([oof, *rest])
    return d.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("p").cast(pl.Float32))


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


def tfidf_mats(split: str) -> dict:
    """TF-IDF (IDF from the pool) L2-normalised hashed matrices for S1 and pool names (core name) and addresses."""
    import scipy.sparse as sp

    P = config.paths()
    pq = P["parquet"] / split
    out = {}
    for kind, col in (("name", "core1"), ("addr", "addr")):
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", col]).sort("rid")[col].to_list()
        s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", col]).sort("rid")[col].to_list()
        s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", col]).sort("rid")[col].to_list()
        pool = _hash_docs(s2 + s3, kind)
        df = np.bincount(pool.indices, minlength=pool.shape[1]).astype(np.float32)
        idf = (np.log((pool.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        for name, m in (("pool", pool), ("s1", _hash_docs(s1, kind))):
            m = m.tocsr()
            m.data *= idf[m.indices]
            norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).astype(np.float32)
            norm[norm == 0] = 1.0
            m = sp.diags(1.0 / norm) @ m
            out[f"{kind}_{name}"] = m.tocsr()
    return out


def tfidf_features(rows: pl.DataFrame, mats: dict, n2: int, step: int = 500_000) -> pl.DataFrame:
    """TF-IDF cosine of S1 and pool name and address for every pair (q, pid)."""
    q = rows["q"].to_numpy()
    pid = rows["pid"].to_numpy()
    pidx = np.where(pid < 3 * PID_BASE, pid - 2 * PID_BASE, n2 + pid - 3 * PID_BASE)
    res = {}
    for kind in ("name", "addr"):
        A, B = mats[f"{kind}_s1"], mats[f"{kind}_pool"]
        v = np.empty(len(q), dtype=np.float32)
        for a in range(0, len(q), step):
            v[a: a + step] = np.asarray(A[q[a: a + step]].multiply(B[pidx[a: a + step]]).sum(axis=1)).ravel()
        res[f"tf_{kind}_cos"] = v
    return pl.DataFrame({"q": q, "pid": pid, **res})


def build(split: str, base: str, prm: dict) -> None:
    """Write consensus-feature chunks for a split under WORK/stack/{split}."""
    P = config.paths()
    t0 = time.time()
    d = load_p1(split, base)
    d = pid_features(d)
    if split == "train":  # keep the locked holdout plus a seeded subsample of the other S1 for training
        hold = holdout_q()
        allq = d.select("q").unique()["q"].to_numpy()
        pool = np.setdiff1d(allq, hold)
        rng = np.random.default_rng(prm["seed"])
        sub = rng.choice(pool, size=min(prm["sub_q"], len(pool)), replace=False)
        keep = np.sort(np.concatenate([hold, sub]))
        d = d.join(pl.DataFrame({"q": keep}), on="q", how="semi")
        feat_files = [str(f) for sub_dir in ("train", "train_rest") for f in sorted((P["work"] / "features" / sub_dir).glob("part_*.parquet"))]
    else:
        keep = np.sort(d.select("q").unique()["q"].to_numpy())
        feat_files = [str(f) for f in sorted((P["work"] / "features" / "test").glob("part_*.parquet"))]
    d = d.sort("q", "pid")
    out = P["work"] / "stack" / split
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    scan = pl.scan_parquet(feat_files)
    s1_addr, pool_addr, n2 = _addr_arrays(split)
    _, pool_name = _name_arrays(split)
    mats = tfidf_mats(split)
    n = 0
    for i, lo_i in enumerate(range(0, len(keep), prm["chunk_q"])):
        qs = keep[lo_i: lo_i + prm["chunk_q"]]
        lo, hi = int(qs[0]), int(qs[-1])
        rows0 = d.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).join(pl.DataFrame({"q": qs}), on="q", how="semi")
        rows = q_features(rows0)
        orig = (scan.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).select(["q", "pid", *ORIG])
                    .with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).collect())
        rows = rows.join(orig, on=["q", "pid"], how="left").join(digit_features(rows0, s1_addr, pool_addr, n2), on=["q", "pid"], how="left")
        rows = rows.join(tfidf_features(rows0, mats, n2), on=["q", "pid"], how="left")
        rows = rows.join(consensus_text_features(rows0, pool_name, pool_addr, n2), on=["q", "pid"], how="left")
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
    return pl.concat([pl.read_parquet(f) for f in sorted((P["work"] / "stack" / split).glob("chunk_*.parquet"))])


def train(name: str, base: str, prm: dict, drop: tuple[str, ...] = ()) -> None:
    """Fit the stacked model on the non-holdout S1, tune the threshold, and score the locked holdout with a paired CI."""
    P = config.paths()
    t0 = time.time()
    df = _read("train")
    hold = pl.DataFrame({"q": holdout_q()})
    is_hold = df.join(hold.with_columns(pl.lit(True).alias("_h")), on="q", how="left")["_h"].fill_null(False).to_numpy()
    feats = [c for c in df.columns if c not in {"q", "pid", "label"} and not c.startswith(drop)]  # `drop`: feature prefixes left out (ablations)
    qa, pida, y = df["q"].to_numpy(), df["pid"].to_numpy(), df["label"].to_numpy()
    x = df.select(feats).to_numpy()  # all Float32: no upcast, one copy
    del df
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
        m = xgb.train(p, xgb.DMatrix(x[tr], label=y[tr], feature_names=feats), prm["rounds"],
                      evals=[(xgb.DMatrix(x[va], label=y[va], feature_names=feats), "val")], early_stopping_rounds=prm["early_stop"], verbose_eval=False)
        oof[va] = m.predict(xgb.DMatrix(x[va], feature_names=feats), iteration_range=(0, m.best_iteration + 1))
        iters.append(m.best_iteration + 1)
        print(f"fold {k}: {iters[-1]} rounds, aucpr {m.best_score:.4f}", flush=True)
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    tune = pl.DataFrame({"q": qa[tr_mask], "pid": pida[tr_mask], "p": oof[tr_mask], "label": y[tr_mask]})
    nt_tune = tune.select("q").unique().join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    thr, score_oof, _ = decision.tune_threshold(tune, nt_tune, True)
    print(f"stacked OOF macro F0.5 on the non-holdout subsample: {score_oof:.4f} at threshold {thr:.2f}", flush=True)
    model = xgb.train(p, xgb.DMatrix(x[tr_mask], label=y[tr_mask], feature_names=feats), int(np.mean(iters) * 1.1) + 1)

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
    (out / "config.json").write_text(json.dumps({"features": feats, "threshold": thr, "exclusive": True, "device_trained": device, "base": base}, indent=2))
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
    model = xgb.Booster()
    model.load_model(str(mdl / "xgb.json"))
    if cfg.get("device_trained") == "cuda":
        model.set_param({"device": "cuda"})
    parts = []
    for f in sorted((P["work"] / "stack" / "test").glob("chunk_*.parquet")):
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
    ap.add_argument("cmd", choices=["build", "train", "predict"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--name", default="s1")
    ap.add_argument("--base", default=None)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to leave out of training (ablation)")
    a = ap.parse_args(argv)
    prm = config.load()["stack"]
    base = a.base or prm["base"]
    if a.cmd == "build":
        build(a.split, base, prm)
    elif a.cmd == "train":
        train(a.name, base, prm, tuple(x for x in a.drop.split(",") if x))
    else:
        predict(a.name)


if __name__ == "__main__":
    main()
