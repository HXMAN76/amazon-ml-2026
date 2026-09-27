import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import synth  # noqa: E402
from ber.stages import block, block_eval, prepare, sample  # noqa: E402


def test_block_and_eval_on_synthetic(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=400, seed=1)
    synth.make(data, "test", n=100, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    monkeypatch.setenv("BER_PARAMS", str(Path(__file__).resolve().parents[2] / "configs" / "params.yaml"))
    prepare.main([])
    sample.main()
    block.main(["--split", "train"])
    block.main(["--split", "test"])
    cand = pl.read_parquet(str(work / "blocks" / "train" / "cand_*.parquet"))
    assert {"q", "pid", "score", "ns", "s_n", "s_a", "s_p", "s_c", "s_m", "s_d", "s_h"} <= set(cand.columns)
    assert cand.group_by("q").len()["len"].max() <= 30
    test_cand = pl.read_parquet(str(work / "blocks" / "test" / "cand_*.parquet"))
    assert test_cand.height > 0
    block_eval.main()
    rep = (work / "blocks" / "eval_report.json").read_text()
    import json

    r = json.loads(rep)
    assert "diagnose" in r and r["diagnose"]["true_pairs"] > 0
    assert r["k30"]["pair_recall"] > 0.9
    assert r["k100"]["pair_recall"] >= r["k30"]["pair_recall"] - 0.001
