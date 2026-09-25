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
    ap.add_argument("--extra", type=int, default=0, help="add this many S1 from the rest (features/train_rest) to the training set")
    ap.add_argument("--set", default="", help="comma-separated parameter overrides, e.g. max_depth=9,eta=0.05,rounds=3000")
    a = ap.parse_args(argv)
    P, prm = config.paths(), dict(config.load()["train_gpu"])
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        prm[k] = type(prm[k])(v)
    t0 = time.time()
    df = pl.read_parquet(str(P["work"] / "features" / "train" / "part_*.parquet"))
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid", "fold", "n_matches"]).rename({"rid": "q"})
    df = df.join(smp.select("q", "fold"), on="q", how="left")
    train_q = smp["q"].to_numpy().astype(np.int64)
    extra_q = np.zeros(0, dtype=np.int64)
    if a.extra:  # more training S1 from the rest: never the holdout, never the S1 the dense encoders were fitted on (their dense features are optimistic)
        from ber.split import holdout_q
        from ber.stages.dense import dense_train_q
        from ber.stages.dense_all import dense_all_train_q

        cfg_all = config.load()
        s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
        excl = np.concatenate([train_q, holdout_q().astype(np.int64), dense_train_q(cfg_all["dense"]), dense_all_train_q(cfg_all["dense_all"])])
        ok = np.setdiff1d(s1, excl)
        extra_q = np.sort(np.random.default_rng(prm["seed"] + 7).choice(ok, size=min(a.extra, len(ok)), replace=False))
        rest = (pl.scan_parquet(str(P["work"] / "features" / "train_rest" / "part_*.parquet")).filter(pl.col("q").is_in(extra_q)).collect())
        fold_e = ((rest["q"].to_numpy().astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(prm["folds"])).astype(np.int64)
        df = pl.concat([df.with_columns(pl.col("fold").cast(pl.Int64)), rest.with_columns(pl.Series("fold", fold_e))], how="diagonal_relaxed")
        train_q = np.concatenate([train_q, extra_q])
        print(f"extra training S1: {len(extra_q)} ({rest.height} pairs)", flush=True)
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
    qs = pl.DataFrame({"q": train_q}).join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
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
    np.save(out / "extra_q.npy", extra_q)  # S1 beyond the sample that the model was trained on (score_rest leaves them out)
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
