"""Label-free France audit of output files. Usage: python src/scripts/france_audit.py RUN [RUN ...]
For each RUN (WORK/output/RUN/matching_results.tsv) the predicted pairs are split into cells: relation kind of the core names (band_kinds.kinds,
plus exact core) x address relation (exact: house_eq > 0.5 and addr_tset >= 90; empty pool address; other) x source (S2/S3). Per cell the pairs
per 1,000 S1 of France are compared with the US and India rates of the same file: the labelled holdout keeps these kinds 99%+ true, so a French
excess over max(US, India) is counted as decoys and a shortfall under min(US, India) as missed copies (both conservative). They are rolled into
an estimated French precision, recall and F0.5 (pair level, with the US/India recall loss of 2.5% added to the truth, minus 0.003 for the
per-S1 macro average as on the holdout), printed per run with the cells that carry the most excess and deficit."""

import sys

import polars as pl

from band_kinds import kinds
from ber import config

PID_BASE = 10_000_000


def load(P):
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "name1", "addr"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"),
        pl.col("name1").alias("b_name"), (pl.col("addr") == "").alias("b_empty"), pl.lit(s).alias("src")) for s in (2, 3)])
    feats = pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet"))).select(
        pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset")
    return s1, pool, feats


def cells(run: str, P, s1, pool, feats) -> pl.DataFrame:
    m = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
    pairs = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .rename({"source1_entity_id": "e1", "matched_entity_ids": "eb"}))
    d = pairs.join(s1, on="e1").join(pool, on="eb")
    d = d.join(feats.join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect(), on=["q", "pid"], how="left")
    d = kinds(d).with_columns(pl.when(pl.col("a_core") == pl.col("b_core")).then(pl.lit("exact_core")).otherwise(pl.col("kind")).alias("kind"),
                              pl.when(pl.col("b_empty")).then(pl.lit("empty")).when((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
                              .then(pl.lit("exact_addr")).otherwise(pl.lit("other_addr")).alias("addr"))
    n1 = s1.group_by("ctry").len().rename({"len": "n_s1"})
    t = d.group_by("ctry", "kind", "addr", "src").len().join(n1, on="ctry").with_columns((1000 * pl.col("len") / pl.col("n_s1")).alias("r"))
    return t.pivot(on="ctry", index=["kind", "addr", "src"], values="r").fill_null(0.0)


def estimate(c: pl.DataFrame, n_fr: int) -> dict:
    c = c.with_columns((pl.col("france") - pl.max_horizontal("us", "india")).clip(lower_bound=0).alias("excess"),
                       (pl.min_horizontal("us", "india") - pl.col("france")).clip(lower_bound=0).alias("deficit"))
    k = n_fr / 1000
    npred, dec, miss = float(c["france"].sum()) * k, float(c["excess"].sum()) * k, float(c["deficit"].sum()) * k
    tp = npred - dec
    truth = (tp + miss) / 0.975
    p, r = tp / npred, tp / truth
    f = 1.25 * p * r / (0.25 * p + r) - 0.003
    return {"pairs": int(npred), "est_decoys": int(dec), "est_missed": int(miss), "precision": round(p, 4), "recall": round(r, 4), "est_france_f05": round(f, 4), "cells": c}


def main() -> None:
    P = config.paths()
    s1, pool, feats = load(P)
    n_fr = s1.filter(pl.col("ctry") == "france").height
    rows = []
    for run in sys.argv[1:]:
        if not (P["work"] / "output" / run / "matching_results.tsv").exists():
            print(f"{run}: no matching_results.tsv", flush=True)
            continue
        e = estimate(cells(run, P, s1, pool, feats), n_fr)
        c = e.pop("cells")
        rows.append({"run": run, **e})
        with pl.Config(tbl_rows=12, tbl_width_chars=200):
            print(f"\n== {run}: {e}")
            print("largest excess:"); print(c.sort("excess", descending=True).head(8))
            print("largest deficit:"); print(c.sort("deficit", descending=True).head(6))
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
