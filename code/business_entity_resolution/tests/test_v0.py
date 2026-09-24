import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
