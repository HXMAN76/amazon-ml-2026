"""score_rest: score the train S1 that are OUTSIDE the training sample with the final model.

These S1 were never used to train the model, so their probabilities are unbiased. Two uses:
  1. a locked holdout (150k S1 drawn once with a fixed seed) with a bootstrap confidence interval for macro F_0.5;
  2. full-density probabilities p1 for every train S1 (test has all S1), needed by record-level and consensus layers.
Inputs : WORK/features/train_rest/part_*.parquet (from `ber.stages.pairs --split train --rest`), WORK/models/<name>/
Outputs: WORK/models/<name>/p1_rest/part_*.parquet (q, pid, p, label), WORK/models/<name>/holdout.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.tracking import log_stage


def main(argv: list[str] | None = None) -> None:
    """CLI: predict the rest of the train S1, report holdout macro F_0.5 with a 95% bootstrap interval."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2")
    ap.add_argument("--holdout", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models" / a.name
    cfg = json.loads((mdl / "config.json").read_text())
    model = xgb.Booster()
    model.load_model(str(mdl / "xgb.json"))
    if cfg.get("device_trained") == "cuda":
        model.set_param({"device": "cuda"})
    feats = cfg["features"]
    out = mdl / "p1_rest"
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for f in sorted((P["work"] / "features" / "train_rest").glob("part_*.parquet")):
        d = pl.read_parquet(f)
        pp = model.predict(xgb.DMatrix(d.select(feats).to_numpy().astype(np.float32), feature_names=feats))
        r = d.select("q", "pid", "label").with_columns(pl.Series("p", pp))
        r.write_parquet(out / f.name, compression="zstd")
        res.append(r)
    df = pl.concat(res)

    sample_q = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"]).rename({"rid": "q"})
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"]).rename({"rid": "q"})
    rest = s1.join(sample_q, on="q", how="anti")
    rng = np.random.default_rng(a.seed)
    hold_q = rest["q"].to_numpy()[np.sort(rng.choice(rest.height, size=min(a.holdout, rest.height), replace=False))]
    hold = pl.DataFrame({"q": hold_q})
    n_true = (pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len()
                .rename({"s1_rid": "q", "len": "n_true"}))
    nt = hold.join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    d = df.join(hold, on="q", how="semi")
    sel = decision.assign_exclusive(d) if cfg["exclusive"] else d
    pred = sel.filter(pl.col("p") >= cfg["threshold"])
    pe = decision.per_entity_f05(pred, nt)
    mean, lo, hi = decision.bootstrap_ci(pe["f"].to_numpy())
    rep = {"model": a.name, "holdout_s1": hold.height, "macro_f05": mean, "ci95": [lo, hi], "threshold": cfg["threshold"],
           "precision": float(pred["label"].mean()) if pred.height else None,
           "recall": float(pred["label"].sum() / max(int(nt["n_true"].sum()), 1))}
    (mdl / "holdout.json").write_text(json.dumps(rep, indent=2))
    print("HOLDOUT (disjoint from the training sample):", json.dumps(rep), flush=True)
    log_stage("score_rest", {"name": a.name, "holdout": a.holdout}, {"holdout_f05": mean, "ci_lo": lo, "ci_hi": hi,
                                                                       "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
