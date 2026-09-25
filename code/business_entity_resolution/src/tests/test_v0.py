import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import decision, synth  # noqa: E402
from ber.stages import block, pairs, predict, prepare, sample, train_gpu  # noqa: E402
from ber.validate import validate  # noqa: E402
import polars as pl  # noqa: E402


def test_decision_metric_matches_worked_example():
    pred = pl.DataFrame({"q": [1, 1, 1], "label": [1, 0, 1]})
    n_true = pl.DataFrame({"q": [1], "n_true": [2]})
    assert abs(decision.macro_f05(pred, n_true) - 0.714) < 1e-3
    # singleton handling: empty prediction on a singleton scores 1, any prediction scores 0
    n2 = pl.DataFrame({"q": [1, 2], "n_true": [0, 0]})
    assert decision.macro_f05(pl.DataFrame({"q": [2], "label": [0]}), n2) == 0.5


def test_v0_end_to_end_on_synthetic(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=500, seed=1)
    synth.make(data, "test", n=200, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    for split in ("train", "test"):
        block.main(["--split", split])
        pairs.main(["--split", split, "--chunk", "3000"])
    train_gpu.main(["--name", "t"])
    predict.main(["--name", "t"])
    out = work / "output" / "t"
    assert validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", data / "test") == []
    assert (work / "models" / "t" / "report.json").exists()


def test_submission_checker_accepts_valid_and_rejects_quoted_empties(tmp_path, monkeypatch):
    from scripts import check_submission

    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=300, seed=1)
    synth.make(data, "test", n=120, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    for split in ("train", "test"):
        block.main(["--split", split])
        pairs.main(["--split", split, "--chunk", "3000"])
    train_gpu.main(["--name", "t"])
    predict.main(["--name", "t"])
    out = work / "output" / "t"
    res = check_submission.check(out, data / "test", verbose=False)
    assert res["issues"] == [], res["issues"]
    assert "france" in {k.lower() for k in res["stats"]["countries"]}
    # the bug that once cost a validation run: an empty list written as a quoted empty string must be flagged
    bad = (out / "matching_results.tsv").read_text().replace("\t\n", '\t""\n')
    (out / "matching_results.tsv").write_text(bad)
    assert check_submission.check(out, data / "test", verbose=False)["issues"]


def test_holdout_scoring_on_the_rest_of_train(tmp_path, monkeypatch):
    import json

    import yaml

    from ber.stages import score_rest

    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=500, seed=1)
    synth.make(data, "test", n=60, countries=("US", "India", "France"), seed=2)
    prm = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "params.yaml").read_text())
    prm["sample"]["n_s1"] = 300  # 300 sampled S1, 200 left outside for the holdout
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(prm))
    monkeypatch.setenv("BER_PARAMS", str(tmp_path / "params.yaml"))
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    block.main(["--split", "train", "--all-train"])
    block.main(["--split", "test"])
    pairs.main(["--split", "train", "--chunk", "3000"])
    pairs.main(["--split", "train", "--rest", "--chunk", "3000"])
    train_gpu.main(["--name", "t"])
    score_rest.main(["--name", "t", "--holdout", "150"])
    rep = json.loads((work / "models" / "t" / "holdout.json").read_text())
    assert rep["holdout_s1"] == 150 and rep["ci95"][0] <= rep["macro_f05"] <= rep["ci95"][1] and rep["macro_f05"] > 0.6


def test_consensus_stacking_end_to_end(tmp_path, monkeypatch):
    import json

    import yaml

    from ber.stages import score_rest, stack

    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=500, seed=1)
    synth.make(data, "test", n=80, countries=("US", "India", "France"), seed=2)
    prm = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "params.yaml").read_text())
    prm["sample"]["n_s1"] = 300
    prm["stack"].update({"base": "t", "sub_q": 250, "chunk_q": 120, "folds": 3, "rounds": 40, "early_stop": 10, "max_depth": 4})
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(prm))
    monkeypatch.setenv("BER_PARAMS", str(tmp_path / "params.yaml"))
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    block.main(["--split", "train", "--all-train"])
    block.main(["--split", "test"])
    pairs.main(["--split", "train", "--chunk", "3000"])
    pairs.main(["--split", "train", "--rest", "--chunk", "3000"])
    pairs.main(["--split", "test", "--chunk", "3000"])
    train_gpu.main(["--name", "t"])
    score_rest.main(["--name", "t"])
    predict.main(["--name", "t"])
    stack.main(["build", "--split", "train"])
    stack.main(["build", "--split", "test"])
    stack.main(["train", "--name", "s"])
    stack.main(["predict", "--name", "s"])
    rep = json.loads((work / "models" / "s" / "holdout.json").read_text())
    assert rep["delta_ci95"][0] <= rep["delta"] <= rep["delta_ci95"][1] and rep["stack_holdout_f05"] > 0.5
    prm["expf"].update({"base": "s", "tune_q": 100, "mu_grid": [0.0, 0.2], "shift_grid": [0.0], "J": 6})
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(prm))
    from ber.stages import expf

    expf.main(["fit", "--base", "s", "--name", "e"])
    expf.main(["predict", "--base", "s", "--name", "e"])
    erep = json.loads((work / "models" / "e" / "holdout.json").read_text())
    assert erep["delta_ci95"][0] <= erep["delta"] <= erep["delta_ci95"][1] and erep["expf_holdout_f05"] > 0.5
    from scripts import check_submission as _chk

    assert _chk.check(work / "output" / "e", data / "test", verbose=False)["issues"] == []
    out = work / "output" / "s"
    from scripts import check_submission

    assert check_submission.check(out, data / "test", verbose=False)["issues"] == []


def test_expected_f05_matches_brute_force():
    import itertools
    import math

    import numpy as np

    from ber.stages import expf

    rng = np.random.default_rng(0)
    P = np.sort(rng.uniform(0.02, 0.98, size=(6, 4)), axis=1)[:, ::-1].copy()
    lam = np.array([0.0, 0.0, 0.3, 0.3, 0.6, 1.0])
    got = expf.expected_f05_all(P, lam)
    for r in range(P.shape[0]):
        for k in range(5):
            exp = 0.0
            for out in itertools.product([0, 1], repeat=4):
                pr = np.prod([P[r, i] if o else 1 - P[r, i] for i, o in enumerate(out)])
                tp = sum(out[:k])
                m_in = sum(out[k:])
                for extra in range(40):
                    pm = math.exp(-lam[r]) * lam[r] ** extra / math.factorial(extra)
                    t_all = tp + m_in + extra
                    f = 1.0 if (k == 0 and t_all == 0) else (0.0 if k == 0 else 1.25 * tp / (0.25 * t_all + k))
                    exp += pr * pm * f
            assert abs(exp - got[r, k]) < 1e-6, (r, k, exp, got[r, k])


def test_isotonic_is_monotone_and_reduces_calibration_error():
    import numpy as np

    from ber.stages import expf

    rng = np.random.default_rng(1)
    true_p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < true_p).astype(int)
    over = np.clip(true_p * 1.3, 0, 1)  # an over-confident score
    knots = expf.fit_isotonic(over, y)
    assert np.all(np.diff(knots[1]) >= -1e-12)
    assert expf.ece(expf.apply_isotonic(over, knots), y) < expf.ece(over, y)


def test_digit_relation_features_flag_dropped_and_added_trailing_digits():
    import numpy as np

    from ber.stages import stack

    s1_addr = np.array(["1223 park avenue", "6252 golden hook"], dtype=object)
    pool = np.array(["122 park ave", "1223 park ave", "12234 park avenue", "625 golden hook", "6252 golden hook"], dtype=object)
    rows = pl.DataFrame({"q": [0, 0, 0, 1, 1], "pid": [20_000_000 + i for i in range(5)], "p": [0.9, 0.8, 0.7, 0.9, 0.8]})
    f = stack.digit_features(rows, s1_addr, pool, n2=5).sort("pid")
    g = {c: f[c].to_list() for c in f.columns}
    assert g["h_b_prefix_of_a"][0] == 1.0 and g["h_eq"][0] == 0.0          # 122 lost the trailing digit of 1223
    assert g["h_eq"][1] == 1.0 and g["h_a_prefix_of_b"][2] == 1.0          # equal; 12234 gained a digit
    assert g["h_b_prefix_of_a"][3] == 1.0 and g["h_eq"][4] == 1.0          # second S1: 625 vs 6252, then equal
    assert g["h_eq_best_other"][1] == 0.0 and g["h_prefix_best_other"][1] == 1.0  # vs best other (122): a prefix relation


def test_dense_topk_and_merge_add_missing_pairs(tmp_path, monkeypatch):
    import numpy as np

    from ber.stages import dense

    rng = np.random.default_rng(0)
    s = rng.normal(size=(40, 16)).astype(np.float32)
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    pool = (s[[3, 7, 11]] + 0.05 * rng.normal(size=(3, 16))).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    got = dense.topk_pairs(s.astype(np.float16), np.arange(100, 140), pool.astype(np.float16), np.array([20_000_001, 20_000_002, 20_000_003]), k=3, device="cpu")
    top1 = got.filter(pl.col("rank") == 0).sort("pid")["q"].to_list()
    assert top1 == [103, 107, 111]

    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=200, seed=1)
    synth.make(data, "test", n=40, seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    block.main(["--split", "train", "--all-train"])
    shards = sorted((work / "blocks" / "train").glob("cand_*.parquet"))
    cand = pl.concat([pl.read_parquet(f) for f in shards])
    before = cand.height
    have = set(zip(cand["q"].to_list(), cand["pid"].to_list()))
    extra = [(int(q), 30_000_000 + 5000 + i) for i, q in enumerate(cand["q"].unique().to_list()[:6])]  # pairs the blocker did not propose
    dp = pl.DataFrame({"q": [e[0] for e in extra] + [int(cand["q"][0])], "pid": [e[1] for e in extra] + [int(cand["pid"][0])],
                       "cos": [0.9] * 7, "rank": [0] * 7}).with_columns(pl.col("rank").cast(pl.Int16))
    (work / "dense" / "train").mkdir(parents=True)
    dp.write_parquet(work / "dense" / "train" / "pairs.parquet")
    dense.merge("train")
    merged = pl.concat([pl.read_parquet(f) for f in sorted((work / "blocks" / "train").glob("cand_*.parquet"))])
    assert merged.height == before + 6 and {"emb_cos", "emb_rank"} <= set(merged.columns)
    assert merged.filter(pl.col("emb_cos").is_not_null()).height == 7
    dense.merge("train")  # idempotent: always merges from the untouched shards
    assert pl.concat([pl.read_parquet(f) for f in sorted((work / "blocks" / "train").glob("cand_*.parquet"))]).height == before + 6
