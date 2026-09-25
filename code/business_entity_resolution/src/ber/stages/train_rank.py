"""train_rank: XGBoost listwise LTR (rank:pairwise, qid=q) as an alternative to train_gpu's pointwise classifier.

Same inputs/outputs/features/folds as train_gpu, only the objective differs. Rationale: the competition metric
(macro F0.5 per S1) rewards getting the *ordering within each S1's candidate group* right, which is exactly
what a listwise ranker optimises directly, instead of a pointwise classifier plus a global threshold. v0's
`margin_p`/`rank_p` features already smuggle within-group competition into the pointwise model; this makes the
training objective itself group-aware, on the same feature set, so it is a fair apples-to-apples comparison
via `decision.tune_threshold` and the paired holdout gate, not a new pipeline.

Inputs : WORK/features/train/part_*.parquet, WORK/sample/train_s1.parquet (folds), labels.parquet
Outputs: WORK/models/<name>/{xgb.json,config.json,report.json,oof.parquet}
Falls back to CPU when no GPU is visible (`train_rank.device: auto`).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.stages.train_gpu import NON_FEATURES, pick_device
from ber.tracking import log_stage


def _grouped_dmatrix(x: np.ndarray, y: np.ndarray, q: np.ndarray, feats: list[str]) -> tuple[xgb.DMatrix, np.ndarray]:
    """XGBoost ranking needs rows sorted by group (qid) and group sizes via `set_group`.

    Returns the DMatrix plus the permutation used, so callers can scatter predictions back to original order.
    """
    order = np.argsort(q, kind="stable")
    qs = q[order]
    _, sizes = np.unique(qs, return_counts=True)  # np.unique returns groups in sorted order, matching `order`
    d = xgb.DMatrix(x[order], label=y[order], feature_names=feats)
    d.set_group(sizes)
    return d, order


def main(argv: list[str] | None = None) -> None:
    """CLI: grouped 5-fold listwise XGBoost ranker, threshold tuning on out-of-fold scores, final model save."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rank0")
    ap.add_argument("--objective", default="rank:pairwise", choices=["rank:pairwise", "rank:ndcg", "rank:map"])
    ap.add_argument("--baseline", default=None, help="name of a train_gpu/train_rank run to compare against (oof.parquet)")
    a = ap.parse_args(argv)
    P, prm = config.paths(), config.load()["train_rank"]
    t0 = time.time()
    df = pl.read_parquet(str(P["work"] / "features" / "train" / "part_*.parquet"))
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid", "fold", "n_matches"]).rename({"rid": "q"})
    df = df.join(smp.select("q", "fold"), on="q", how="left")
    feats = [c for c in df.columns if c not in NON_FEATURES]
    device = pick_device(prm["device"])
    print(f"{df.height} pairs, {len(feats)} features, positives {df['label'].sum()}, device={device}, objective={a.objective}", flush=True)

    x = df.select(feats).to_numpy().astype(np.float32)
    y = df["label"].to_numpy()
    q = df["q"].to_numpy()
    fold = df["fold"].to_numpy()
    p = {"objective": a.objective, "eval_metric": prm["eval_metric"], "device": device, "tree_method": "hist",
         "max_depth": prm["max_depth"], "eta": prm["eta"], "subsample": prm["subsample"],
         "colsample_bytree": prm["colsample"], "min_child_weight": prm["min_child_weight"], "seed": prm["seed"],
         "lambdarank_num_pair_per_sample": prm.get("num_pair_per_sample", 8)}
    oof = np.zeros(len(y), dtype=np.float32)
    best_iters = []
    for k in range(prm["folds"]):
        tr, va = fold != k, fold == k
        dtr, otr = _grouped_dmatrix(x[tr], y[tr], q[tr], feats)
        dva, ova = _grouped_dmatrix(x[va], y[va], q[va], feats)
        t = time.time()
        m = xgb.train(p, dtr, prm["rounds"], evals=[(dva, "val")], early_stopping_rounds=prm["early_stop"], verbose_eval=False)
        pred_va = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        va_idx = np.flatnonzero(va)
        oof[va_idx[ova]] = pred_va  # scatter back through both the boolean mask and the group sort permutation
        best_iters.append(m.best_iteration + 1)
        print(f"fold {k}: {best_iters[-1]} rounds, {prm['eval_metric']} {m.best_score:.4f} in {time.time() - t:.0f}s", flush=True)

    # ranker scores are relative within a group, not calibrated probabilities: min-max per S1 before thresholding,
    # so decision.tune_threshold's fixed global grid (0.10..0.95) is meaningful the same way it is for train_gpu.
    oof_df = df.select("q", "pid", "label").with_columns(pl.Series("raw", oof))
    oof_df = oof_df.with_columns(
        ((pl.col("raw") - pl.col("raw").min().over("q")) /
         (pl.col("raw").max().over("q") - pl.col("raw").min().over("q")).clip(lower_bound=1e-6)).alias("p"))
    n_true = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename(
        {"s1_rid": "q", "len": "n_true"})
    qs = smp.select("q").join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    res = {}
    for excl in (False, True):
        thr, score, curve = decision.tune_threshold(oof_df.select("q", "pid", "p", "label"), qs, excl)
        res["exclusive" if excl else "plain"] = {"threshold": thr, "macro_f05": score}
        print(f"OOF macro F0.5 ({'exclusive' if excl else 'plain'}): {score:.4f} at threshold {thr:.2f}", flush=True)
    best_mode = max(res, key=lambda k: res[k]["macro_f05"])

    compare = {}
    if a.baseline:
        base_oof = pl.read_parquet(P["work"] / "models" / a.baseline / "oof.parquet")
        base_cfg = json.loads((P["work"] / "models" / a.baseline / "config.json").read_text())
        base_sel = decision.assign_exclusive(base_oof) if base_cfg["exclusive"] else base_oof
        base_score = decision.macro_f05(base_sel.filter(pl.col("p") >= base_cfg["threshold"]), qs)
        delta = res[best_mode]["macro_f05"] - base_score
        pe_new = decision.per_entity_f05(
            (decision.assign_exclusive(oof_df) if best_mode == "exclusive" else oof_df).filter(pl.col("p") >= res[best_mode]["threshold"]), qs)
        pe_base = decision.per_entity_f05(base_sel.filter(pl.col("p") >= base_cfg["threshold"]), qs)
        # paired bootstrap on the per-entity delta (same S1 set both sides), not two independent bootstraps
        merged = pe_new.rename({"f": "f_new"}).join(pe_base.rename({"f": "f_base"}), on="q")
        d = (merged["f_new"] - merged["f_base"]).to_numpy()
        mean, lo, hi = decision.bootstrap_ci(d)
        compare = {"baseline": a.baseline, "baseline_macro_f05": base_score, "delta": delta,
                   "paired_delta_ci95": [lo, hi], "ship": lo > 0}
        print(f"vs {a.baseline}: delta {delta:+.4f}, paired 95% CI [{lo:+.4f}, {hi:+.4f}], ship={lo > 0}", flush=True)

    final_rounds = int(np.mean(best_iters) * 1.1) + 1
    dfull, ofull = _grouped_dmatrix(x, y, q, feats)
    model = xgb.train(p, dfull, final_rounds)
    out = P["work"] / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "xgb.json"))
    cfg = {"features": feats, "threshold": res[best_mode]["threshold"], "exclusive": best_mode == "exclusive",
           "rounds": final_rounds, "device_trained": device, "objective": a.objective, "ranker": True}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "report.json").write_text(json.dumps({**res, "best": best_mode, "rounds": best_iters, "compare": compare}, indent=2))
    oof_df.select("q", "pid", "p", "label").write_parquet(out / "oof.parquet", compression="zstd")
    imp = model.get_score(importance_type="gain")
    print("top features:", sorted(imp.items(), key=lambda kv: -kv[1])[:10], flush=True)
    log_stage("train_rank", prm, {"oof_f05_plain": res["plain"]["macro_f05"], "oof_f05_exclusive": res["exclusive"]["macro_f05"],
                                   "threshold": cfg["threshold"], "seconds": time.time() - t0, **compare})


if __name__ == "__main__":
    main()
