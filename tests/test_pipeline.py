import json

import numpy as np
import polars as pl
import pytest

from amlc.evaluation import extraction_f1, smape
from amlc.features.text import extract_field, format_value_unit, parse_value_unit
from amlc.inference.shard import merge_parts, part_path, shard_bounds
from amlc.models.gbm import blend, cv_train
from amlc.submission.validator import SubmissionSpec, validate


def test_shard_bounds_cover_all_rows_once():
    n, k = 1003, 7
    rows = [r for i in range(k) for r in range(*shard_bounds(n, i, k))]
    assert rows == list(range(n))


def test_merge_parts_detects_gaps_and_overlaps(tmp_path):
    for s, e in [(0, 5), (5, 10)]:
        pl.DataFrame({"row": list(range(s, e)), "y": [0.0] * (e - s)}).write_parquet(part_path(tmp_path, s, e))
    assert merge_parts(tmp_path, expected_rows=10).height == 10
    with pytest.raises(ValueError, match="expected 12"):
        merge_parts(tmp_path, expected_rows=12)
    pl.DataFrame({"row": [4], "y": [1.0]}).write_parquet(part_path(tmp_path, 4, 5))
    with pytest.raises(ValueError, match="duplicated"):
        merge_parts(tmp_path)


def test_metrics():
    assert smape([100, 0], [100, 0]) == 0
    assert smape([100], [50]) == pytest.approx(100 * 50 / 75)
    # tp=1, fp=1 (wrong value), fn=1 (empty pred)
    assert extraction_f1(["1 gram", "2 gram", "3 gram"], ["1 gram", "5 gram", ""]) == pytest.approx(0.5)


def test_value_unit_parsing():
    assert parse_value_unit("Net wt 500 g pack") == (500.0, "gram")
    assert parse_value_unit("12.5 Fl Oz bottle") == (12.5, "fluid ounce")
    assert parse_value_unit("weight 2,5 kg") == (2.5, "kilogram")
    assert parse_value_unit("5 cm and 3 kg", allowed_units={"kilogram"}) == (3.0, "kilogram")
    assert parse_value_unit("no numbers") is None
    assert format_value_unit(34.50, "gram") == "34.5 gram"
    assert extract_field("Item Name: Tea Value: 12.0 Unit: Ounce", "Value") == "12.0"
    assert extract_field("Item Name: Tea Value: 12.0 Unit: Ounce", "Unit") == "Ounce"


def test_validator():
    spec = SubmissionSpec(id_col="sample_id", pred_cols=["price"], min_value=0)
    good = pl.DataFrame({"sample_id": ["1", "2", "3"], "price": ["1.5", "2", "0"]})
    assert validate(good, spec, test_ids=[1, 2, 3]) == []

    bad = pl.DataFrame({"sample_id": ["1", "1", "9"], "price": ["-1", "nan", "abc"]})
    errs = " | ".join(validate(bad, spec, test_ids=[1, 2, 3]))
    for frag in ["duplicated", "missing", "not in test", "non-numeric", "NaN", "< 0"]:
        assert frag in errs, frag

    spec2 = SubmissionSpec(id_col="index", pred_cols=["prediction"], pred_type="str", allow_empty=True,
                           regex=r"^$|^\d+(\.\d+)? (gram|kilogram)$")
    sub = pl.DataFrame({"index": ["0", "1", "2"], "prediction": ["1.5 gram", "", "2 grams"]})
    assert "format regex" in " ".join(validate(sub, spec2))


def test_cv_train_and_blend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=400)) * 10
    Xt = rng.normal(size=(50, 5))
    r1 = cv_train(X, y, Xt, model="lgbm", metric="smape", target="log1p", n_folds=3, name="a",
                  params={"n_estimators": 200})
    r2 = cv_train(X, y, Xt, model="xgb", metric="smape", target="log1p", n_folds=3, name="b",
                  params={"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "colsample_bytree": 1.0,
                                    "early_stopping_rounds": 20})
    assert r1["score"] < 30 and r2["score"] < 30
    b = blend(["a", "b"], y, metric="smape")
    assert b["score"] <= min(r1["score"], r2["score"]) + 1e-6
    assert abs(sum(b["weights"].values()) - 1) < 1e-9


def test_tracking_writes_runs_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import amlc.tracking as tr

    monkeypatch.setattr(tr, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    with tr.track("EXP-TEST", params={"lr": 0.1}, tags={"dataset": "v1"}) as run:
        run.log_metrics({"cv": 1.23})
    rec = json.loads((tmp_path / "runs.jsonl").read_text())
    assert rec["metrics"]["cv"] == 1.23 and rec["status"] == "ok" and rec["tags"]["dataset"] == "v1"


@pytest.mark.parametrize("task", ["reg", "clf"])
def test_baseline_end_to_end(tmp_path, monkeypatch, task):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    words = ["tea", "coffee", "sugar", "pack", "bottle", "organic", "large", "small"]
    n = 300

    def rows(k, offset):
        qty = rng.integers(1, 50, k)
        text = [f"Item Name: {' '.join(rng.choice(words, 4))} Value: {q} Unit: Ounce" for q in qty]
        return pl.DataFrame({"sample_id": np.arange(offset, offset + k), "catalog_content": text,
                             "price": qty * 2.0 + rng.normal(0, 1, k), "label": np.where(qty > 25, "big", "small")})

    tr, te = rows(n, 0), rows(60, 10_000).drop("price", "label")
    tr.write_csv(tmp_path / "train.csv")
    te.write_csv(tmp_path / "test.csv")
    target, pred_type = ("price", "float") if task == "reg" else ("label", "str")
    (tmp_path / "spec.yaml").write_text(f"id_col: sample_id\npred_cols: [{target}]\npred_type: {pred_type}\n")

    from amlc.baseline import main

    main(["--train", "train.csv", "--test", "test.csv", "--id-col", "sample_id", "--target", target,
          "--text-cols", "catalog_content", "--task", task, "--metric", "smape" if task == "reg" else "accuracy",
          "--svd-dim", "8", "--folds", "3", "--spec", "spec.yaml", "--out", "sub.csv"])
    sub = pl.read_csv("sub.csv")
    assert sub.height == 60 and sub.columns == ["sample_id", target]
    if task == "clf":
        assert set(sub[target].unique().to_list()) <= {"big", "small"}
