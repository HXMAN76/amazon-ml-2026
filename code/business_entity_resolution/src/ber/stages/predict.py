"""predict: score the test candidates, decide, write output/{matching_results,candidate_pairs}.tsv and validate.

Inputs : WORK/features/test/part_*.parquet, WORK/models/<name>/, WORK/parquet/test/source*.parquet
Outputs: WORK/output/<name>/{matching_results.tsv,candidate_pairs.tsv}
The official utils/validate_submission.py is used when found at WORK/official/validate_submission.py,
otherwise the local re-implementation in ber.validate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

from ber import config, decision
from ber.stages.block import PID_BASE
from ber.tracking import log_stage
from ber.validate import validate


def main(argv: list[str] | None = None) -> None:
    """CLI: score test candidates, apply the decision rule, write both TSV outputs and validate them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v0")
    a = ap.parse_args(argv)
    P = config.paths()
    t0 = time.time()
    mdl = P["work"] / "models" / a.name
    cfg = json.loads((mdl / "config.json").read_text())
    model = xgb.Booster()
    model.load_model(str(mdl / "xgb.json"))
    model.set_param({"device": "cuda"} if cfg.get("device_trained") == "cuda" else {})
    feats = cfg["features"]
    res = []
    for f in sorted((P["work"] / "features" / "test").glob("part_*.parquet")):  # score part by part: 52M rows do not fit RAM at once
        d = pl.read_parquet(f)
        pp = model.predict(xgb.DMatrix(d.select(feats).to_numpy().astype(np.float32), feature_names=feats))
        res.append(d.select("q", "pid").with_columns(pl.Series("p", pp)))
    df = pl.concat(res)
    p = df["p"].to_numpy()
    (P["work"] / "output" / a.name).mkdir(parents=True, exist_ok=True)
    df.write_parquet(P["work"] / "output" / a.name / "pair_p.parquet", compression="zstd")  # p1 of every test candidate pair (q, pid, p)
    emit(a.name, P, df, cfg, t0)


def emit(name: str, P: dict, df: pl.DataFrame, cfg: dict, t0: float) -> None:
    """Apply the decision rule to scored pairs (q, pid, p), write both TSVs and validate them."""
    p = df["p"].to_numpy()
    sel = decision.assign_exclusive(df) if cfg["exclusive"] else df
    sel = sel.filter(pl.col("p") >= cfg["threshold"])
    print(f"{df.height} candidate pairs scored, {sel.height} kept at threshold {cfg['threshold']:.2f} "
          f"(exclusive={cfg['exclusive']}), mean p {float(p.mean()):.4f}", flush=True)

    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id"]).sort("rid")
    s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", "entity_id"])
    s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", "entity_id"])
    pool = pl.concat([s2.with_columns((pl.col("rid").cast(pl.Int64) + 2 * PID_BASE).alias("pid")),
                      s3.with_columns((pl.col("rid").cast(pl.Int64) + 3 * PID_BASE).alias("pid"))]).select("pid", pid_eid="entity_id")

    def lists(d: pl.DataFrame, col: str, step: int = 200_000) -> pl.DataFrame:
        """Per-S1 comma-joined id lists, built in q-ranges to bound memory."""
        d = d.sort("q")
        parts = []
        for lo in range(0, s1.height, step):
            g = (d.filter((pl.col("q") >= lo) & (pl.col("q") < lo + step)).join(pool, on="pid")
                   .group_by("q").agg(pl.col("pid_eid").sort().str.join(",").alias(col)))
            parts.append(g)
        g = pl.concat(parts) if parts else pl.DataFrame({"q": [], col: []})
        return (s1.rename({"rid": "q", "entity_id": "source1_entity_id"}).join(g, on="q", how="left")
                  .with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col))

    out = P["work"] / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    m = lists(sel, "matched_entity_ids")
    c = lists(df, "candidate_entity_ids")
    m.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")  # never quote: an empty list must be an empty field, not ""
    c.write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    n_match = int((m["matched_entity_ids"] != "").sum())
    print(f"wrote {out}: {m.height} S1 rows, {n_match} with matches ({n_match / m.height:.1%}), "
          f"mean matches {float(sel.height / m.height):.2f}", flush=True)

    official = P["work"] / "official" / "validate_submission.py"
    if official.exists():
        r = subprocess.run([sys.executable, str(official), "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(P["data"] / "test")],
                           capture_output=True, text=True)
        print("OFFICIAL VALIDATOR:", r.stdout.strip()[-600:], r.stderr.strip()[-300:], flush=True)
    else:
        issues = validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", P["data"] / "test")
        print("LOCAL VALIDATOR:", "PASS" if not issues else issues[:10], flush=True)
    log_stage("predict", {"name": name}, {"pairs": float(df.height), "kept": float(sel.height),
                                             "matched_s1_frac": n_match / m.height, "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
