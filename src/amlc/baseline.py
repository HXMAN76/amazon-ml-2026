"""One-command text baseline to get a valid, timestamped submission in the first hours.

TF-IDF(word+char) -> SVD + numeric text stats (+ any numeric columns) -> k-fold GBM -> validated CSV.

    python -m amlc.baseline --train data/raw/train.csv --test data/raw/test.csv \
        --id-col sample_id --target price --text-cols catalog_content --metric smape --target-tf log1p \
        --out outputs/sub_baseline.csv

Classification: add --task clf (labels are encoded and decoded automatically).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from amlc.data.io import read_table
from amlc.features.text import numeric_stats, tfidf_svd
from amlc.models.gbm import cv_train
from amlc.submission.validator import SubmissionSpec, validate
from amlc.tracking import track


def build_features(train: pl.DataFrame, test: pl.DataFrame, text_cols: list[str], id_col: str, target: str,
                   svd_dim: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    def joined(df):
        return df.select(pl.concat_str([pl.col(c).cast(pl.String).fill_null("") for c in text_cols],
                                       separator=" | ")).to_series().to_list()

    tr_txt, te_txt = joined(train), joined(test)
    Ztr, Zte = tfidf_svd(tr_txt, te_txt, n_components=svd_dim)
    parts_tr, parts_te = [Ztr, numeric_stats(tr_txt)], [Zte, numeric_stats(te_txt)]
    names = [f"svd{i}" for i in range(svd_dim)] + ["n_num", "num_max", "num_min", "num_first", "len", "words"]

    skip = {id_col, target, *text_cols}
    num_cols = [c for c, t in test.schema.items() if c not in skip and t.is_numeric()]
    if num_cols:
        parts_tr.append(train.select(num_cols).to_numpy().astype(np.float32))
        parts_te.append(test.select(num_cols).to_numpy().astype(np.float32))
        names += num_cols
    return np.hstack(parts_tr), np.hstack(parts_te), names


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--id-col", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--text-cols", nargs="+", required=True)
    p.add_argument("--task", choices=["reg", "clf"], default="reg")
    p.add_argument("--metric", default="rmse")
    p.add_argument("--target-tf", choices=["none", "log1p"], default="none")
    p.add_argument("--model", choices=["lgbm", "xgb", "cat"], default="lgbm")
    p.add_argument("--svd-dim", type=int, default=256)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--clip-min", type=float, default=None, help="clip regression preds, e.g. 0 for prices")
    p.add_argument("--spec", default="configs/submission.yaml")
    p.add_argument("--out", default="outputs/sub_baseline.csv")
    p.add_argument("--name", default="baseline-tfidf")
    a = p.parse_args(argv)

    train, test = read_table(a.train), read_table(a.test)
    train = train.filter(pl.col(a.target).is_not_null())
    X, Xt, _names = build_features(train, test, a.text_cols, a.id_col, a.target, a.svd_dim)
    y = train[a.target].to_numpy()
    classes = None
    if a.task == "clf":
        classes, y = np.unique(y, return_inverse=True)
    print(f"features {X.shape}, test {Xt.shape}")

    params = {"n_folds": a.folds, "model": a.model, "metric": a.metric, "task": a.task}
    with track(a.name, params=params, tags={"features": "tfidf-svd+numstats"}) as run:
        r = cv_train(X, y, Xt, model=a.model, metric=a.metric, task=a.task,
                     target=None if a.target_tf == "none" else a.target_tf, n_folds=a.folds, name=a.name)
        run.log_metrics({f"cv_{a.metric}": r["score"]})

    pred = r["test"]
    if a.task == "clf":
        pred = classes[pred.argmax(1)]
    elif a.clip_min is not None:
        pred = np.maximum(pred, a.clip_min)

    spec = SubmissionSpec.from_yaml(a.spec)
    sub = pl.DataFrame({spec.id_col: test[a.id_col], spec.pred_cols[0]: pred})
    errs = validate(sub.cast(pl.String), spec, test[a.id_col].to_list())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(a.out)
    print(("INVALID: " + "; ".join(errs)) if errs else f"OK submission -> {a.out}")


if __name__ == "__main__":
    main()
