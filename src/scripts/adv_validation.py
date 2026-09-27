"""Adversarial validation: which first-stage features tell a labelled holdout pair from a test pair?

Usage: python src/scripts/adv_validation.py
Takes pairs of the locked holdout S1 (features/train_rest) and of the test S1 of the same countries (US and India; France has no train
counterpart), trains an XGBoost classifier for "is a test pair" on the first-stage features and prints the AUC, the most important
features and, for each of them, the standardised mean difference between test and holdout. A high AUC means the test set looks
different from the data used for evaluation; the top features are the candidates to normalise or to drop.
"""

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ber import config
from ber.split import holdout_q

NON = {"q", "pid", "label", "fold"}


def main() -> None:
    P = config.paths()
    hq = holdout_q()
    tr = pl.concat([pl.read_parquet(f).filter(pl.col("q").is_in(hq)) for f in sorted((P["work"] / "features" / "train_rest").glob("part_*.parquet"))])
    te = pl.concat([pl.read_parquet(f) for f in sorted((P["work"] / "features" / "test").glob("part_*.parquet"))])
    s1te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"})
    te = te.join(s1te, on="q", how="left").filter(pl.col("ctry").is_in(["us", "india"])).drop("ctry")
    feats = [c for c in tr.columns if c not in NON and c in te.columns]
    rng = np.random.default_rng(0)
    n = min(1_500_000, tr.height, te.height)
    a = tr.select(feats).sample(n, seed=1).to_numpy().astype(np.float32)
    b = te.select(feats).sample(n, seed=2).to_numpy().astype(np.float32)
    x = np.vstack([a, b])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    idx = rng.permutation(len(y))
    x, y = x[idx], y[idx]
    cut = int(0.8 * len(y))
    p = {"objective": "binary:logistic", "eval_metric": "auc", "device": "cuda", "tree_method": "hist", "max_depth": 6, "eta": 0.1}
    d1, d2 = xgb.DMatrix(x[:cut], label=y[:cut], feature_names=feats), xgb.DMatrix(x[cut:], label=y[cut:], feature_names=feats)
    m = xgb.train(p, d1, 200, evals=[(d2, "val")], verbose_eval=50)
    auc = roc_auc_score(y[cut:], m.predict(d2))
    print(f"adversarial AUC (holdout pair versus test pair, US and India): {auc:.4f} on {n} pairs each", flush=True)
    imp = m.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:20]
    print("feature | gain | mean holdout | mean test | standardised difference")
    for f, g in top:
        i = feats.index(f)
        ma, mb = float(np.nanmean(a[:, i])), float(np.nanmean(b[:, i]))
        sd = float(np.nanstd(np.concatenate([a[:, i], b[:, i]]))) or 1.0
        print(f"{f:22s} {g:10.1f} {ma:10.4f} {mb:10.4f} {(mb - ma) / sd:+.3f}", flush=True)
    for grp, cols in {"S1/pool counts": [c for c in feats if c.startswith(("log_cnt", "n_q_for_p", "n_cand_q"))]}.items():
        for c in cols:
            i = feats.index(c)
            print(f"[{grp}] {c:18s} holdout {float(np.nanmean(a[:, i])):8.4f} test {float(np.nanmean(b[:, i])):8.4f}")


if __name__ == "__main__":
    main()
