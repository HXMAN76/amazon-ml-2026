"""Holdout score and threshold re-weighted to the test set's mix of countries and look-alike density (label-free).

Usage: python src/scripts/test_weighted.py MODEL [--apply]      (MODEL: a stacked model, e.g. bs_s6)

The test has about twice as many ownerless look-alike records per S1 as train (density_check.py: they are decoys, not orphans),
47% India instead of 40%, and 15% France with no labels, so the plain holdout overstates the leaderboard and its threshold is tuned
for train's decoy density. Every S1 falls into a cell (country, u) where u = its number of plausible-but-weak candidates
(0.05 <= p1 < 0.5 among the model's shortlist, capped at 4), which is computed the same way on test without labels. The estimate
is post-stratified: the test's share of each cell times the holdout's mean F0.5 in that cell (France uses the pooled US and India
S1 with the same u). The threshold is re-tuned for the same weighting on the out-of-fold S1, never on the holdout.
--apply writes models/MODELtw (the same model with the re-tuned threshold) and its test output through predict.emit.
"""

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from ber.stages.stack import load_p1

U_LO, U_HI, U_CAP = 0.05, 0.5, 4


def cells(rows: pl.DataFrame, p1: pl.DataFrame, ctry: pl.DataFrame) -> pl.DataFrame:
    """(q, c, u) for the S1 of `rows` (the model's shortlist pairs), with p1 from the first stage."""
    u = (rows.select("q", "pid").join(p1, on=["q", "pid"], how="left")
             .group_by("q").agg(((pl.col("p1") >= U_LO) & (pl.col("p1") < U_HI)).sum().clip(upper_bound=U_CAP).alias("u")))
    return u.join(ctry, on="q", how="left")


def per_s1(pred: pl.DataFrame, thr: float, nt: pl.DataFrame, cap: dict) -> pl.DataFrame:
    sel = pred.filter(pl.col("p") >= thr)
    return decision.per_entity_f05(decision.cap_per_source(sel, cap) if cap else sel, nt)


def weighted(f: pl.DataFrame, c: pl.DataFrame, share: pl.DataFrame) -> float:
    """Post-stratified mean: sum over test cells of share x holdout mean F in the cell (France and empty cells: pooled by u)."""
    d = f.join(c, on="q", how="inner")
    by_cu = d.group_by("c", "u").agg(pl.col("f").mean().alias("fc"))
    by_u = d.group_by("u").agg(pl.col("f").mean().alias("fu"))
    t = share.join(by_cu, on=["c", "u"], how="left").join(by_u, on="u", how="left")
    return float(t.select((pl.col("share") * pl.coalesce("fc", "fu")).sum()).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    P = config.paths()
    t0 = time.time()
    M = P["work"] / "models" / a.name
    cfg = json.loads((M / "config.json").read_text())
    cap = {int(k): int(v) for k, v in (cfg.get("cap") or {}).items()}
    base = cfg["base"]
    ids = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))  # noqa: E731
    ctry = lambda split: pl.read_parquet(P["parquet"] / split / "source1.parquet", columns=["rid", "ctry"]).select(  # noqa: E731
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("ctry").cast(pl.Utf8).alias("c"))
    p1_tr = ids(load_p1("train", base)).select("q", "pid", pl.col("p").alias("p1"))
    p1_te = ids(pl.read_parquet(P["work"] / "output" / base / "pair_p.parquet")).select("q", "pid", pl.col("p").alias("p1"))
    test = ids(pl.read_parquet(P["work"] / "output" / a.name / "pair_p.parquet"))
    tc = cells(test, p1_te, ctry("test"))
    tc = ctry("test").join(tc.select("q", "u"), on="q", how="left").with_columns(pl.col("u").fill_null(0))
    share = tc.group_by("c", "u").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("share")).drop("len")

    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().select(
        pl.col("s1_rid").cast(pl.Int64).alias("q"), pl.col("len").alias("n_true"))
    rep = {"model": a.name, "threshold": cfg["threshold"], "test_cells": share.sort("c", "u").to_dicts()}
    for split, f in (("oof", "oof_tune.parquet"), ("holdout", "holdout_pred.parquet")):
        pred = decision.assign_exclusive(ids(pl.read_parquet(M / f)))
        qs = pl.DataFrame({"q": holdout_q().astype(np.int64)}) if split == "holdout" else pred.select("q").unique()  # every holdout S1, even without candidates
        c = qs.join(ctry("train"), on="q", how="left").join(cells(pred, p1_tr, ctry("train")).select("q", "u"), on="q", how="left").with_columns(
            pl.col("u").fill_null(0))
        nt = qs.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
        if split == "oof":  # threshold re-tuned for the test's weighting, out of fold only
            t0_ = round(float(cfg["threshold"]), 2)  # search around the model's own threshold, which is always in the grid
            grid = np.unique(np.round(np.concatenate([np.arange(max(0.05, t0_ - 0.3), min(0.97, t0_ + 0.3) + 1e-9, 0.01), [t0_]]), 2))
            curve = [(float(t), weighted(per_s1(pred, t, nt, cap), c, share)) for t in grid]
            thr_w = max(curve, key=lambda x: x[1])[0]
            rep["oof_weighted_curve"] = {f"{t:.2f}": round(v, 5) for t, v in curve}
            rep["threshold_weighted"] = thr_w
        for label, thr in (("model_threshold", cfg["threshold"]), ("weighted_threshold", thr_w)):
            f_s1 = per_s1(pred, thr, nt, cap)
            rep[f"{split}_{label}"] = {"plain": float(f_s1["f"].mean()), "test_weighted": weighted(f_s1, c, share)}
    out = P["work"] / "measure"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"test_weighted_{a.name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("test_cells", "oof_weighted_curve")}, indent=2))
    h = rep["holdout_model_threshold"]
    print(f"TESTWEIGHTED {a.name}: holdout plain {h['plain']:.5f}, test-weighted {h['test_weighted']:.5f} at threshold {cfg['threshold']:.2f}; "
          f"re-tuned threshold {thr_w:.2f} -> test-weighted {rep['holdout_weighted_threshold']['test_weighted']:.5f} "
          f"(plain {rep['holdout_weighted_threshold']['plain']:.5f})", flush=True)
    if a.apply:
        from ber.stages.predict import emit

        name = a.name + "tw"
        (P["work"] / "models" / name).mkdir(parents=True, exist_ok=True)
        cfg_w = {**cfg, "threshold": thr_w, "threshold_source": f"test_weighted {a.name}"}
        (P["work"] / "models" / name / "config.json").write_text(json.dumps(cfg_w, indent=2))
        (P["work"] / "output" / name).mkdir(parents=True, exist_ok=True)
        test.write_parquet(P["work"] / "output" / name / "pair_p.parquet", compression="zstd")
        emit(name, P, test.select("q", "pid", "p"), cfg_w, t0)


if __name__ == "__main__":
    main()
