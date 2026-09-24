"""prepare: raw TSV -> normalised Parquet (per split and source) plus a label table.

Outputs under WORK/parquet/:
  {split}/source{1,2,3}.parquet   rid, entity_id, country, business_name, business_address + text.COLUMNS
  train/labels.parquet            s1_rid, src (2 or 3), other_rid   (one row per true pair)
  meta.json                       row counts, checks
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import polars as pl

from ber import config, text
from ber.tracking import log_stage


def _work(rows: list[tuple[str, str, str]]) -> list[tuple]:
    return [text.normalise_row(*r) for r in rows]


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: the files are plain tab-separated; a stray quote in a name must not swallow rows
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.all().fill_null("")
    )


def prepare_source(tsv: Path, out: Path, workers: int, chunk_rows: int) -> int:
    df = read_tsv(tsv)
    df = df.with_row_index("rid")
    names, addrs, ctry = df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()
    chunks = [
        list(zip(names[s : s + chunk_rows], addrs[s : s + chunk_rows], ctry[s : s + chunk_rows]))
        for s in range(0, len(names), chunk_rows)
    ]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_work, chunks)
    cols = list(zip(*[row for part in parts for row in part]))
    derived = pl.DataFrame(
        {c: list(v) for c, v in zip(text.COLUMNS, cols)},
        schema={"name1": pl.Utf8, "name2": pl.Utf8, "core1": pl.Utf8, "legal": pl.Utf8, "is_domain": pl.Boolean,
                "has_alias": pl.Boolean, "addr": pl.Utf8, "ctry": pl.Utf8, "nl_name": pl.Float32, "nl_addr": pl.Float32},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.hstack(derived).write_parquet(out, compression="zstd")
    return df.height


def build_labels(raw_gt: Path, pq: Path) -> pl.DataFrame:
    s1 = pl.read_parquet(pq / "train" / "source1.parquet", columns=["rid", "entity_id"])
    oth = pl.concat([pl.read_parquet(pq / "train" / f"source{i}.parquet", columns=["rid", "entity_id"]).with_columns(
        pl.lit(i, dtype=pl.UInt8).alias("src")) for i in (2, 3)])
    gt = read_tsv(raw_gt)
    long = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    lab = (long.join(s1.rename({"rid": "s1_rid", "entity_id": "source1_entity_id"}), on="source1_entity_id", how="left")
               .join(oth.rename({"rid": "other_rid", "entity_id": "matched_entity_ids"}), on="matched_entity_ids", how="left"))
    missing = lab.filter(pl.col("s1_rid").is_null() | pl.col("other_rid").is_null()).height
    assert missing == 0, f"{missing} ground-truth ids not found in the source files"
    return lab.select("s1_rid", "src", "other_rid")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    a = ap.parse_args(argv)
    P, prm = config.paths(), config.load()["prepare"]
    splits = ["train", "test"] if a.split == "both" else [a.split]
    meta: dict = {}
    t0 = time.time()
    for sp in splits:
        for i in (1, 2, 3):
            t = time.time()
            n = prepare_source(P["data"] / sp / f"{sp}_source{i}.tsv", P["parquet"] / sp / f"source{i}.parquet",
                               prm["workers"], prm["chunk_rows"])
            meta[f"{sp}_source{i}"] = n
            print(f"{sp} source{i}: {n} rows in {time.time() - t:.0f}s", flush=True)
    if "train" in splits:
        lab = build_labels(P["data"] / "train" / "train_ground_truth.tsv", P["parquet"])
        lab.write_parquet(P["parquet"] / "train" / "labels.parquet", compression="zstd")
        meta["train_pairs"] = lab.height
        print(f"labels: {lab.height} true pairs", flush=True)
    (P["parquet"] / "meta.json").write_text(json.dumps(meta, indent=2))
    log_stage("prepare", prm, {**{k: float(v) for k, v in meta.items()}, "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
