"""K-fold GBM trainer (LightGBM / XGBoost / CatBoost) with OOF + test predictions,
and an OOF-based blender. OOF/test preds are saved so any member can ensemble later.

    res = cv_train(X, y, X_test, model="lgbm", metric="smape", target="log1p", name="EXP-003")
    blend(["EXP-003", "EXP-007"], y, metric="smape")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from amlc.evaluation import get_metric

PRED_DIR = Path("artifacts/preds")

DEFAULTS = {
    "lgbm": dict(n_estimators=5000, learning_rate=0.03, num_leaves=127, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1),
    "xgb": dict(n_estimators=5000, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                tree_method="hist", early_stopping_rounds=200),
    "cat": dict(iterations=5000, learning_rate=0.05, depth=8, verbose=0, early_stopping_rounds=200),
}

TARGETS = {
    None: (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),  # right-skewed targets like price: optimises ~relative error
}


def _make(model: str, task: str, params: dict, gpu: bool):
    p = {**DEFAULTS[model], **params}
    if model == "lgbm":
        import lightgbm as lgb

        if gpu:
            p.setdefault("device_type", "gpu")
        return (lgb.LGBMRegressor if task == "reg" else lgb.LGBMClassifier)(**p)
    if model == "xgb":
        import xgboost as xgb

        if gpu:
            p.setdefault("device", "cuda")
        return (xgb.XGBRegressor if task == "reg" else xgb.XGBClassifier)(**p)
    import catboost as cb

    if gpu:
        p.setdefault("task_type", "GPU")
    return (cb.CatBoostRegressor if task == "reg" else cb.CatBoostClassifier)(**p)


def _fit(est, model, Xtr, ytr, Xva, yva):
    if model == "lgbm":
        import lightgbm as lgb

        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(200, verbose=False)])
    elif model == "xgb":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        est.fit(Xtr, ytr, eval_set=(Xva, yva))
    return est


def cv_train(X, y, X_test=None, model: str = "lgbm", metric: str = "rmse", task: str = "reg",
             target: str | None = None, n_folds: int = 5, seed: int = 42, params: dict | None = None,
             gpu: bool = False, name: str | None = None) -> dict:
    """Returns {"oof", "test", "fold_scores", "score"}; for task="clf" oof/test are class probabilities
    (labels 0..k-1 = argmax). Saves preds under artifacts/preds/<name>/."""
    fwd, inv = TARGETS[target]
    y = np.asarray(y)
    yt = fwd(y) if task == "reg" else y
    metric_fn, _ = get_metric(metric)
    splitter = (StratifiedKFold if task == "clf" else KFold)(n_folds, shuffle=True, random_state=seed)
    # classification keeps class probabilities (averaged over folds); labels = argmax
    n_cls = len(np.unique(y)) if task == "clf" else 0
    oof = np.zeros((len(y), n_cls) if n_cls else len(y), dtype=float)
    test = None
    if X_test is not None:
        test = np.zeros((len(X_test), n_cls) if n_cls else len(X_test), dtype=float)
    scores = []
    for k, (tr, va) in enumerate(splitter.split(X, y)):
        est = _fit(_make(model, task, params or {}, gpu), model, X[tr], yt[tr], X[va], yt[va])
        if task == "reg":
            oof[va] = inv(est.predict(X[va]))
            scores.append(metric_fn(y[va], oof[va]))
        else:
            oof[va] = est.predict_proba(X[va])
            scores.append(metric_fn(y[va], oof[va].argmax(1)))
        print(f"  fold {k}: {metric}={scores[-1]:.5f}", flush=True)
        if X_test is not None:
            test += (inv(est.predict(X_test)) if task == "reg" else est.predict_proba(X_test)) / n_folds
    score = metric_fn(y, oof if task == "reg" else oof.argmax(1))
    print(f"CV {metric}={score:.5f} (folds {np.mean(scores):.5f} +- {np.std(scores):.5f})")
    if name:
        d = PRED_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "oof.npy", oof)
        if test is not None:
            np.save(d / "test.npy", test)
    return {"oof": oof, "test": test, "fold_scores": scores, "score": score}


def blend(names: list[str], y, metric: str = "rmse") -> dict:
    """Non-negative weights summing to 1 that optimise the metric on saved regression OOFs."""
    from scipy.optimize import minimize

    fn, greater = get_metric(metric)
    oofs = np.stack([np.load(PRED_DIR / n / "oof.npy") for n in names], 1)
    tests = np.stack([np.load(PRED_DIR / n / "test.npy") for n in names], 1)
    sign = -1 if greater else 1

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return sign * fn(y, oofs @ w)

    w0 = np.full(len(names), 1 / len(names))
    w = np.abs(minimize(loss, w0, method="Nelder-Mead").x)
    w /= w.sum()
    score = fn(y, oofs @ w)
    print("blend weights", dict(zip(names, np.round(w, 3))), f"{metric}={score:.5f}")
    return {"weights": dict(zip(names, w)), "oof": oofs @ w, "test": tests @ w, "score": score}
