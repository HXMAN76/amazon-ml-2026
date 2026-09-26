import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import decision  # noqa: E402
from ber.stages.xenc import siblings, with_sibling  # noqa: E402

S2, S3 = 20_000_000, 30_000_000


def test_sibling_prefers_confident_other_source_record():
    p1 = pl.DataFrame({"q": [1, 1, 1, 1, 2, 2], "pid": [S2 + 1, S2 + 2, S3 + 1, S3 + 2, S2 + 5, S3 + 5],
                       "p": [0.9, 0.3, 0.8, 0.95, 0.99, 0.2]})
    sib = {(q, p): s for q, p, s in siblings(p1).iter_rows()}
    assert sib[(1, S2 + 2)] == S3 + 2  # other source first, its most confident record
    assert sib[(1, S3 + 1)] == S2 + 1  # S3 pair -> best S2 record
    assert sib[(1, S3 + 2)] == S2 + 1
    assert sib[(2, S3 + 5)] == S2 + 5
    assert sib[(2, S2 + 5)] is None  # its only other-source candidate is below 0.5, no same-source alternative
    p1b = pl.DataFrame({"q": [3, 3, 3], "pid": [S2 + 7, S2 + 8, S2 + 9], "p": [0.9, 0.7, 0.1]})
    sib = {(q, p): s for q, p, s in siblings(p1b).iter_rows()}
    assert sib[(3, S2 + 7)] == S2 + 8 and sib[(3, S2 + 8)] == S2 + 7 and sib[(3, S2 + 9)] == S2 + 7  # same source, never itself
    assert with_sibling({3: "a"}, {S2 + 8: "b"}, 3, S2 + 8) == "a [SIB] b" and with_sibling({3: "a"}, {}, 3, None) == "a"


def test_cap_keeps_most_confident_per_source():
    sel = pl.DataFrame({"q": [1] * 7, "pid": [S2 + i for i in range(7)], "p": [0.9, 0.8, 0.7, 0.99, 0.6, 0.95, 0.65]})
    kept = decision.cap_per_source(sel, {2: 5, 3: 6})
    assert kept.height == 5 and set(kept["p"].to_list()) == {0.99, 0.95, 0.9, 0.8, 0.7}


def _toy_stack(work: Path, n_q: int, seed: int) -> None:
    """Stack artifacts for a toy task only sibling attention solves: a candidate is true iff its f1 is the largest of its S1's,
    while p_stack (per row) is uninformative."""
    import json

    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        f1 = rng.normal(size=4)
        for j in range(4):
            rows.append((q, S2 + q * 10 + j, int(f1[j] == f1.max()), float(f1[j]), float(rng.normal())))
    d = pl.DataFrame(rows, schema=["q", "pid", "label", "f1", "f2"], orient="row").with_columns(pl.col("label").cast(pl.Int8))
    for split in ("train", "test"):
        (work / "stack" / split).mkdir(parents=True)
        (d if split == "train" else d.drop("label")).write_parquet(work / "stack" / split / "chunk_0000.parquet")
    m = work / "models" / "stk"
    m.mkdir(parents=True)
    (m / "config.json").write_text(json.dumps({"features": ["f1", "f2"], "tag": "", "threshold": 0.5, "cap": {}}))
    ps = d.select("q", "pid", "label", pl.lit(0.5).alias("p"))
    ps.filter(pl.col("q") >= n_q // 5).write_parquet(m / "oof_tune.parquet")
    ps.filter(pl.col("q") < n_q // 5).write_parquet(m / "holdout_pred.parquet")
    (work / "output" / "stk").mkdir(parents=True)
    ps.select("q", "pid", "p").write_parquet(work / "output" / "stk" / "pair_p.parquet")
    (work / "parquet" / "train").mkdir(parents=True)
    d.filter(pl.col("label") == 1).select(pl.col("q").alias("s1_rid"), pl.lit(2).alias("src"), (pl.col("pid") - S2).alias("other_rid")).write_parquet(
        work / "parquet" / "train" / "labels.parquet")


def test_set_model_learns_from_siblings_and_blend_ships(tmp_path, monkeypatch):
    import numpy as np

    from ber.stages import predict, setmodel

    n_q = 1500
    _toy_stack(tmp_path, n_q, seed=0)
    monkeypatch.setenv("BER_WORK", str(tmp_path))
    monkeypatch.setattr(setmodel, "holdout_q", lambda: np.arange(n_q // 5))
    monkeypatch.setattr(setmodel, "final_q", lambda: np.arange(n_q // 5, n_q // 5 + 100))  # scored, never trained on
    monkeypatch.setattr(setmodel, "PRM", {**setmodel.PRM, "epochs": 12, "batch": 64, "d": 32, "layers": 2, "folds": 3})
    setmodel.train("setm", "stk")
    hold = pl.read_parquet(tmp_path / "models" / "setm" / "holdout_pred.parquet")
    top = hold.sort("p", descending=True).group_by("q").first()
    assert top["label"].mean() > 0.9  # picks the sibling with the largest f1, which no single row can tell
    emitted = {}
    monkeypatch.setattr(predict, "emit", lambda name, P, df, cfg, t0: emitted.update(name=name, cfg=cfg, n=df.height))
    setmodel.blend("final", "stk", "setm")
    rep = __import__("json").loads((tmp_path / "models" / "final" / "holdout.json").read_text())
    assert rep["ship"] and rep["blend_holdout_f05"] > rep["stack_holdout_f05"] + 0.3
    assert emitted["name"] == "final" and emitted["n"] == 4 * n_q and emitted["cfg"]["weight_stack"] < 1
    fin = pl.read_parquet(tmp_path / "models" / "setm" / "final_pred.parquet")
    oof = pl.read_parquet(tmp_path / "models" / "setm" / "oof.parquet")
    assert fin["q"].n_unique() == 100 and oof.join(fin, on=["q", "pid"], how="semi").height == 0  # final S1 never in training


def test_density_mixture_recovers_orphan_share():
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location("density_check", Path(__file__).resolve().parents[1] / "scripts" / "density_check.py")
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    rng = np.random.default_rng(0)
    decoy, orphan = rng.normal(0.93, 0.03, 20000), rng.normal(0.72, 0.06, 20000)
    target = np.concatenate([rng.normal(0.93, 0.03, 6000), rng.normal(0.72, 0.06, 4000)])  # 40% orphans
    assert abs(dc.mixture_weight(target, orphan, decoy) - 0.4) <= 0.03
