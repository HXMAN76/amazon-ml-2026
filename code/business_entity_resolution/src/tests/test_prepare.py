import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import synth  # noqa: E402
from ber.data import read_tsv  # noqa: E402
from ber.stages import prepare, sample  # noqa: E402


def test_prepare_and_sample(tmp_path, monkeypatch):
    data, work = tmp_path / "dataset", tmp_path / "work"
    synth.make(data, "train", n=200, seed=1)
    synth.make(data, "test", n=80, countries=("US", "India", "France"), seed=2)
    monkeypatch.setenv("BER_DATA", str(data))
    monkeypatch.setenv("BER_WORK", str(work))
    prepare.main([])
    pq = work / "parquet"
    s1 = pl.read_parquet(pq / "train" / "source1.parquet")
    assert s1.height == len(read_tsv(data / "train" / "train_source1.tsv"))
    assert {"rid", "entity_id", "name1", "core1", "legal", "addr", "ctry", "nl_name"} <= set(s1.columns)
    assert s1["rid"].to_list() == list(range(s1.height))
    gt = read_tsv(data / "train" / "train_ground_truth.tsv")
    n_pairs = sum(len([x for x in s.split(",") if x]) for s in gt["matched_entity_ids"])
    lab = pl.read_parquet(pq / "train" / "labels.parquet")
    assert lab.height == n_pairs and set(lab["src"].unique().to_list()) <= {2, 3}
    assert (pq / "test" / "source3.parquet").exists()
    sample.main()
    smp = pl.read_parquet(work / "sample" / "train_s1.parquet")
    assert smp.height == s1.height and set(smp["fold"].unique().to_list()) <= set(range(5))
    assert (work / "runs" / "runs.jsonl").exists()
