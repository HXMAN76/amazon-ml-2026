"""Re-emit a model's output with per-country thresholds (the pair probabilities are unchanged).

Usage: python src/scripts/reemit.py NAME NEWNAME [--thr france=0.85,us=0.71,india=0.71] [--global-thr 0.71]
Reads WORK/output/NAME/pair_p.parquet and models/NAME/config.json, keeps the exclusive assignment of the model, applies the country's
threshold (default: the model's threshold), and writes WORK/output/NEWNAME/{matching_results,candidate_pairs}.tsv through the normal
`emit` path (validation included). The candidate file is unchanged: only the matches change.
"""

import argparse
import json
import time

import polars as pl

from ber import config
from ber.stages.predict import emit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("newname")
    ap.add_argument("--thr", default="", help="comma-separated country=threshold, e.g. france=0.85")
    ap.add_argument("--global-thr", type=float, default=None)
    a = ap.parse_args()
    P = config.paths()
    t0 = time.time()
    cfg = json.loads((P["work"] / "models" / a.name / "config.json").read_text())
    base = a.global_thr if a.global_thr is not None else cfg["threshold"]
    thr = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.thr.split(",") if kv}
    df = pl.read_parquet(P["work"] / "output" / a.name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    d = df.join(s1, on="q", how="left")
    t_c = pl.col("ctry").replace_strict({c: v for c, v in thr.items()}, default=base, return_dtype=pl.Float64)
    # emit applies one global threshold: map every pair to a probability that is above it exactly when it passes the country's threshold
    g = base
    d = d.with_columns(pl.when(pl.col("p") >= t_c).then(pl.max_horizontal(pl.col("p"), pl.lit(g))).otherwise(pl.min_horizontal(pl.col("p"), pl.lit(g - 1e-6))).alias("p2"))
    out = d.select("q", "pid", pl.col("p2").alias("p"))
    (P["work"] / "output" / a.newname).mkdir(parents=True, exist_ok=True)
    out.write_parquet(P["work"] / "output" / a.newname / "pair_p.parquet", compression="zstd")
    emit(a.newname, P, out, {"threshold": g, "exclusive": cfg["exclusive"]}, t0)
    print(f"re-emitted {a.name} as {a.newname} with thresholds {thr or 'unchanged'} (global {base})", flush=True)


if __name__ == "__main__":
    main()
