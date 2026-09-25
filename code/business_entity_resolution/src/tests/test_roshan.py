import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from ber import decision, synth  # noqa: E402
from ber.stages import block, pairs, predict, prepare, sample, train_gpu, train_rank  # noqa: E402
from ber.validate import validate  # noqa: E402


def test_consensus_prune_drops_conflicting_minority_not_confirming_majority():
    # q=1: two selected members. r1 confirms on postcode (the true match). r2 conflicts on postcode: a
    # look-alike distractor with a changed digit. r2 should be dropped.
    sel = pl.DataFrame({
        "q": [1, 1, 2, 2],
        "pid": [10, 11, 20, 21],
        "p": [0.9, 0.6, 0.7, 0.55],
        "pin_match": [True, False, False, False],
        "pin_conflict": [False, True, False, True],
        "house_eq": [False, False, False, False],
    })
    out = decision.consensus_prune(sel)
    assert out["pid"].to_list() == [10, 20, 21]  # r2 (pid 11) dropped; q=2's majority doesn't confirm, so untouched


def test_consensus_prune_leaves_singletons_and_no_conflict_alone():
    sel = pl.DataFrame({
        "q": [1, 2, 2],
        "pid": [10, 20, 21],
        "p": [0.9, 0.8, 0.7],
        "pin_match": [False, True, False],
        "pin_conflict": [False, False, False],
        "house_eq": [False, False, False],
    })
    out = decision.consensus_prune(sel)
    assert out.height == 3  # nothing conflicts, nothing dropped


def test_consensus_prune_improves_macro_f05_on_a_worked_example():
    n_true = pl.DataFrame({"q": [1], "n_true": [1]})
    sel = pl.DataFrame({
        "q": [1, 1], "pid": [10, 11], "p": [0.9, 0.6], "label": [1, 0],
        "pin_match": [True, False], "pin_conflict": [False, True], "house_eq": [False, False],
    })
    before = decision.macro_f05(sel.select("q", "label"), n_true)
    after = decision.macro_f05(decision.consensus_prune(sel).select("q", "label"), n_true)
    assert after > before
    assert after == 1.0  # the distractor is gone, the true match alone is a perfect prediction


def test_train_rank_end_to_end_on_synthetic(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=600, seed=1)
    synth.make(data, "test", n=200, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    for split in ("train", "test"):
        block.main(["--split", split])
        pairs.main(["--split", split, "--chunk", "3000"])
    train_gpu.main(["--name", "base"])
    train_rank.main(["--name", "rank0", "--baseline", "base"])
    predict.main(["--name", "rank0"])
    out = work / "output" / "rank0"
    assert validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", data / "test") == []
    cfg = (work / "models" / "rank0" / "config.json").read_text()
    assert '"ranker": true' in cfg
    report = (work / "models" / "rank0" / "report.json").read_text()
    assert "compare" in report and "ship" in report


def test_predict_consensus_flag_only_removes_pairs(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=500, seed=3)
    synth.make(data, "test", n=150, countries=("US", "India", "France"), seed=4)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    for split in ("train", "test"):
        block.main(["--split", split])
        pairs.main(["--split", split, "--chunk", "3000"])
    train_gpu.main(["--name", "c0"])
    predict.main(["--name", "c0"])
    plain = pl.read_csv(work / "output" / "c0" / "matching_results.tsv", separator="\t")
    predict.main(["--name", "c0", "--consensus"])
    out = work / "output" / "c0"
    assert validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", data / "test") == []
    pruned = pl.read_csv(out / "matching_results.tsv", separator="\t")
    n_plain = sum(len(x) for x in plain["matched_entity_ids"].fill_null("") if x)
    n_pruned = sum(len(x) for x in pruned["matched_entity_ids"].fill_null("") if x)
    assert n_pruned <= n_plain  # consensus_prune only removes, never adds
