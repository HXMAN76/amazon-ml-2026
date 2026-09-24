"""train_gpu: XGBoost (CUDA) pair classifier, grouped 5-fold OOF, exclusive assignment + threshold tuning.

Inputs : WORK/features/train/part_*.parquet, WORK/sample/train_s1.parquet (folds), labels.parquet
Outputs: WORK/models/<name>/{xgb.json,config.json,report.json,oof.parquet}
Falls back to CPU when no GPU is visible (`train_gpu.device: auto`).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.tracking import log_stage

NON_FEATURES = {"q", "pid", "label", "fold"}


def pick_device(want: str) -> str:
    """Return "cuda" when a GPU can run XGBoost, else "cpu"."""
    if want != "auto":
        return want
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.zeros((4, 1), np.float32), label=[0, 1, 0, 1]), 1)
        return "cuda"
    except Exception:  # noqa: BLE001 - no usable GPU
        return "cpu"


def main(argv: list[str] | None = None) -> None:
    """CLI: grouped 5-fold XGBoost training, threshold tuning on out-of-fold scores, final model save."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v0")
    a = ap.parse_args(argv)
    P, prm = config.paths(), config.load()["train_gpu"]
    t0 = time.time()
    df = pl.read_parquet(str(P["work"] / "features" / "train" / "part_*.parquet"))
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid", "fold", "n_matches"]).rename({"rid": "q"})
    df = df.join(smp.select("q", "fold"), on="q", how="left")
    feats = [c for c in df.columns if c not in NON_FEATURES]
    device = pick_device(prm["device"])
    print(f"{df.height} pairs, {len(feats)} features, positives {df['label'].sum()}, device={device}", flush=True)

    x = df.select(feats).to_numpy().astype(np.float32)
    y = df["label"].to_numpy()
    fold = df["fold"].to_numpy()
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "device": device, "tree_method": "hist",
         "max_depth": prm["max_depth"], "eta": prm["eta"], "subsample": prm["subsample"],
         "colsample_bytree": prm["colsample"], "min_child_weight": prm["min_child_weight"], "seed": prm["seed"]}
    oof = np.zeros(len(y), dtype=np.float32)
    best_iters = []
    for k in range(prm["folds"]):
        tr, va = fold != k, fold == k
        dtr, dva = xgb.DMatrix(x[tr], label=y[tr], feature_names=feats), xgb.DMatrix(x[va], label=y[va], feature_names=feats)
        t = time.time()
        m = xgb.train(p, dtr, prm["rounds"], evals=[(dva, "val")], early_stopping_rounds=prm["early_stop"], verbose_eval=False)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        best_iters.append(m.best_iteration + 1)
        print(f"fold {k}: {best_iters[-1]} rounds, aucpr {m.best_score:.4f} in {time.time() - t:.0f}s", flush=True)

    df = df.with_columns(pl.Series("p", oof))
    n_true = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename(
        {"s1_rid": "q", "len": "n_true"})
    qs = smp.select("q").join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    res = {}
    for excl in (False, True):
        thr, score, curve = decision.tune_threshold(df.select("q", "pid", "p", "label"), qs, excl)
        res["exclusive" if excl else "plain"] = {"threshold": thr, "macro_f05": score}
        print(f"OOF macro F0.5 ({'exclusive' if excl else 'plain'}): {score:.4f} at threshold {thr:.2f}", flush=True)
    best_mode = max(res, key=lambda k: res[k]["macro_f05"])
    final_rounds = int(np.mean(best_iters) * 1.1) + 1
    model = xgb.train(p, xgb.DMatrix(x, label=y, feature_names=feats), final_rounds)
    out = P["work"] / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "xgb.json"))
    cfg = {"features": feats, "threshold": res[best_mode]["threshold"], "exclusive": best_mode == "exclusive",
           "rounds": final_rounds, "device_trained": device}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "report.json").write_text(json.dumps({**res, "best": best_mode, "rounds": best_iters}, indent=2))
    df.select("q", "pid", "p", "label").write_parquet(out / "oof.parquet", compression="zstd")
    imp = model.get_score(importance_type="gain")
    print("top features:", sorted(imp.items(), key=lambda kv: -kv[1])[:10], flush=True)
    log_stage("train_gpu", prm, {"oof_f05_plain": res["plain"]["macro_f05"], "oof_f05_exclusive": res["exclusive"]["macro_f05"],
                                 "threshold": cfg["threshold"], "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
