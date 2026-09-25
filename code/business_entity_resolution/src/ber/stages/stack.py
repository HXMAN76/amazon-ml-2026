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
    n = 0
    for i, lo_i in enumerate(range(0, len(keep), prm["chunk_q"])):
        qs = keep[lo_i: lo_i + prm["chunk_q"]]
        lo, hi = int(qs[0]), int(qs[-1])
        rows0 = d.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).join(pl.DataFrame({"q": qs}), on="q", how="semi")
        rows = q_features(rows0)
        orig = (scan.filter((pl.col("q") >= lo) & (pl.col("q") <= hi)).select(["q", "pid", *ORIG])
                    .with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).collect())
        rows = rows.join(orig, on=["q", "pid"], how="left")
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


def train(name: str, base: str, prm: dict) -> None:
    """Fit the stacked model on the non-holdout S1, tune the threshold, and score the locked holdout with a paired CI."""
    P = config.paths()
    t0 = time.time()
    df = _read("train")
    hold = pl.DataFrame({"q": holdout_q()})
    is_hold = df.join(hold.with_columns(pl.lit(True).alias("_h")), on="q", how="left")["_h"].fill_null(False).to_numpy()
    feats = [c for c in df.columns if c not in {"q", "pid", "label"}]
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
    a = ap.parse_args(argv)
    prm = config.load()["stack"]
    base = a.base or prm["base"]
    if a.cmd == "build":
        build(a.split, base, prm)
    elif a.cmd == "train":
        train(a.name, base, prm)
    else:
        predict(a.name)


if __name__ == "__main__":
    main()
