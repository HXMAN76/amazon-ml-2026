import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import synth  # noqa: E402
from ber.stages import block, pairs, prepare, prune, sample  # noqa: E402


def test_cascade_blocking_keeps_recall_at_smaller_k(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=500, seed=1)
    synth.make(data, "test", n=150, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    sample.main()
    block.main(["--split", "train", "--k", "60", "--out-name", "train_raw"])
    block.main(["--split", "test", "--k", "60", "--out-name", "test_raw"])
    prm = {"k_raw": 60, "k_keep": 15, "rounds": 60, "max_depth": 4, "eta": 0.2, "seed": 0}
    rep = prune.train(k_keep=15, params=prm)
    assert rep["share_of_found_kept_by_model_top15"] >= rep["share_of_found_kept_by_score_top15"] - 0.02
    assert prune.apply("train", 15, prm) > 0 and prune.apply("test", 15, prm) > 0
    cand = pl.read_parquet(str(work / "blocks" / "test" / "cand_*.parquet"))
    assert cand.group_by("q").len()["len"].max() <= 15 and "p_block" in cand.columns
    pairs.main(["--split", "train", "--chunk", "3000"])  # downstream stages accept the pruned shards
    feats = pl.read_parquet(str(work / "features" / "train" / "part_*.parquet"))
    assert "p_block" not in feats.columns
