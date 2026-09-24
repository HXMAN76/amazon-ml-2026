import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import metrics, synth  # noqa: E402
from ber.run import main  # noqa: E402


def test_f05_examples():
    # worked example from the problem statement: 0.714
    assert abs(metrics.f05_entity({"a", "b", "c"}, {"a", "c"}) - 0.714) < 1e-3
    assert metrics.f05_entity(set(), set()) == 1.0
    assert metrics.f05_entity({"a"}, set()) == 0.0
    assert metrics.f05_entity(set(), {"a"}) == 0.0


def test_end_to_end(tmp_path, monkeypatch):
    synth.make(tmp_path, "train", n=300, seed=1)
    synth.make(tmp_path, "test", n=150, countries=("US", "India", "France"), seed=2)
    monkeypatch.setattr(sys, "argv", ["ber", "train", "--data", str(tmp_path), "--model", str(tmp_path / "m"), "--folds", "3"])
    main()
    rep = json.loads((tmp_path / "m" / "report.json").read_text())
    assert rep["blocking"]["candidate_recall"] > 0.95
    assert rep["oof"]["macro_f05"] > 0.8
    monkeypatch.setattr(sys, "argv", ["ber", "predict", "--data", str(tmp_path), "--model", str(tmp_path / "m"), "--out", str(tmp_path / "out")])
    main()
    from ber.validate import validate

    assert validate(tmp_path / "out/matching_results.tsv", tmp_path / "out/candidate_pairs.tsv", tmp_path / "test") == []
