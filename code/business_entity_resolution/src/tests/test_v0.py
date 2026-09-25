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
    out = work / "output" / "s"
    from scripts import check_submission

    assert check_submission.check(out, data / "test", verbose=False)["issues"] == []
