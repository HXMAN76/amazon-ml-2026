"""train_stack: stage 2c stacking matcher (v2.md).

Trains a second-stage XGBoost pointwise classifier on the same 63 pair features as v0base's train_gpu,
plus two additions: `p1` (v0base's own OOF probability) and `xs` (the fine-tuned cross-encoder's score
from train_xenc/score_xenc, present only inside the uncertain band; `has_xs` flags where it applies).
Stacking on top of p1 rather than replacing it lets the model learn when to trust the cross-encoder's
token-level signal versus the base feature classifier, instead of forcing one objective to do both jobs
(the mistake that likely hurt train_rank's listwise objective).

Inputs : WORK/features/train/part_*.parquet (63 features, q, pid, label)
         WORK/models/<baseline>/oof.parquet (q, pid, p) -> p1
         WORK/v2/<xenc_name>/xenc_train.parquet (q, pid, xs), from score_xenc --split train
         WORK/sample/train_s1.parquet (fold)
Outputs: WORK/models/<name>/{xgb.json,config.json,report.json,oof.parquet}
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.stages.train_gpu import NON_FEATURES, pick_device
from ber.tracking import log_stage


def main(argv: list[str] | None = None) -> None:
    """CLI: grouped 5-fold XGBoost stacking on base features + p1 + xs, paired-bootstrap vs baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stack0")
    ap.add_argument("--baseline", default="v0base", help="pair model supplying p1 (also the comparison baseline)")
    ap.add_argument("--xenc-name", default="xenc0", help="run name of a completed score_xenc --split train")
    a = ap.parse_args(argv)
    P, prm = config.paths(), config.load()["train_gpu"]  # reuse train_gpu's hyperparameters
    t0 = time.time()

    df = pl.read_parquet(str(P["work"] / "features" / "train" / "part_*.parquet"))
    p1 = pl.read_parquet(P["work"] / "models" / a.baseline / "oof.parquet").select("q", "pid", pl.col("p").alias("p1"))
    xs_path = P["work"] / "v2" / a.xenc_name / "xenc_train.parquet"
    xs = (pl.read_parquet(xs_path).select("q", "pid", "xs") if xs_path.exists()
          else pl.DataFrame({"q": [], "pid": [], "xs": []}, schema={"q": pl.Int64, "pid": pl.Int64, "xs": pl.Float32}))

    df = df.join(p1, on=["q", "pid"], how="left").join(xs, on=["q", "pid"], how="left")
    df = df.with_columns(
        pl.col("p1").fill_null(0.0),
        pl.col("xs").is_not_null().cast(pl.Float32).alias("has_xs"),
        pl.col("xs").fill_null(0.0),
    )
    smp = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid", "fold"]).rename({"rid": "q"})
    df = df.join(smp, on="q", how="left")

    feats = [c for c in df.columns if c not in NON_FEATURES]
    device = pick_device(prm["device"])
    print(f"{df.height} pairs, {len(feats)} features (base + p1 + xs + has_xs), "
          f"positives {df['label'].sum()}, xs coverage {float(df['has_xs'].mean()):.4f}, device={device}", flush=True)

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
        del dtr, dva, m
        gc.collect()

    df = df.with_columns(pl.Series("p", oof))
    n_true = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename(
        {"s1_rid": "q", "len": "n_true"})
    qs = smp.select("q").join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    res = {}
    for excl in (False, True):
        thr, score, _ = decision.tune_threshold(df.select("q", "pid", "p", "label"), qs, excl)
        res["exclusive" if excl else "plain"] = {"threshold": thr, "macro_f05": score}
        print(f"OOF macro F0.5 ({'exclusive' if excl else 'plain'}): {score:.4f} at threshold {thr:.2f}", flush=True)
    best_mode = max(res, key=lambda k: res[k]["macro_f05"])

    base_oof = pl.read_parquet(P["work"] / "models" / a.baseline / "oof.parquet")
    base_cfg = json.loads((P["work"] / "models" / a.baseline / "config.json").read_text())
    base_sel = decision.assign_exclusive(base_oof) if base_cfg["exclusive"] else base_oof
    base_score = decision.macro_f05(base_sel.filter(pl.col("p") >= base_cfg["threshold"]), qs)
    sel = decision.assign_exclusive(df) if best_mode == "exclusive" else df
    pe_new = decision.per_entity_f05(sel.filter(pl.col("p") >= res[best_mode]["threshold"]), qs)
    pe_base = decision.per_entity_f05(base_sel.filter(pl.col("p") >= base_cfg["threshold"]), qs)
    merged = pe_new.rename({"f": "f_new"}).join(pe_base.rename({"f": "f_base"}), on="q")
    delta_vals = (merged["f_new"] - merged["f_base"]).to_numpy()
    mean, lo, hi = decision.bootstrap_ci(delta_vals)
    compare = {"baseline": a.baseline, "baseline_macro_f05": base_score, "delta": res[best_mode]["macro_f05"] - base_score,
               "paired_delta_ci95": [lo, hi], "ship": lo > 0}
    print(f"vs {a.baseline}: delta {compare['delta']:+.4f}, paired 95% CI [{lo:+.4f}, {hi:+.4f}], ship={lo > 0}", flush=True)

    final_rounds = int(np.mean(best_iters) * 1.1) + 1
    model = xgb.train(p, xgb.DMatrix(x, label=y, feature_names=feats), final_rounds)
    out = P["work"] / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "xgb.json"))
    cfg = {"features": feats, "threshold": res[best_mode]["threshold"], "exclusive": best_mode == "exclusive",
           "rounds": final_rounds, "device_trained": device, "baseline": a.baseline, "xenc_name": a.xenc_name}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "report.json").write_text(json.dumps({**res, "best": best_mode, "rounds": best_iters, "compare": compare}, indent=2))
    df.select("q", "pid", "p", "label").write_parquet(out / "oof.parquet", compression="zstd")
    imp = model.get_score(importance_type="gain")
    print("top features:", sorted(imp.items(), key=lambda kv: -kv[1])[:10], flush=True)
    log_stage("train_stack", prm, {"oof_f05_plain": res["plain"]["macro_f05"], "oof_f05_exclusive": res["exclusive"]["macro_f05"],
                                    "threshold": cfg["threshold"], "seconds": time.time() - t0, **compare})


if __name__ == "__main__":
    main()
