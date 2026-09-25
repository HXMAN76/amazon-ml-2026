"""audit_recall: measure what fraction of true pairs the token-index blocker actually generates.

This is the zero-cost check that v4 (ARCHITECTURE_v4.md) gates dense blocking behind: if token
blocking already recovers ~100% of true pairs, dense blocking has nothing to add and should not
be built. If it's leaking true pairs, that's the number that justifies building it.

Inputs : WORK/blocks/train/cand_*.parquet (q, pid candidates from token blocking, pre-feature)
         DATA/train/labels.parquet (s1_rid, src, other_rid: the true pairs)
Outputs: prints recall = |true pairs present in candidates| / |true pairs|, plus a breakdown by
         source (src=2, src=3) since the two pools can have different recall.
"""

from __future__ import annotations

import time

import polars as pl

from ber import config
from ber.stages.block import PID_BASE
from ber.tracking import log_stage


def main(argv: list[str] | None = None) -> None:
    """CLI: join true pairs against the token-blocked candidate set and report recall."""
    P = config.paths()
    t0 = time.time()

    cand = pl.read_parquet(str(P["work"] / "blocks" / "train" / "cand_*.parquet")).select("q", "pid").unique()
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"),
        pl.col("s1_rid").alias("q"),
    ).select("q", "pid", "src")

    hit = lab.join(cand, on=["q", "pid"], how="semi")
    recall = hit.height / lab.height
    print(f"overall recall: {hit.height}/{lab.height} = {recall:.4%}", flush=True)

    for src in sorted(lab["src"].unique().to_list()):
        sub = lab.filter(pl.col("src") == src)
        sub_hit = sub.join(cand, on=["q", "pid"], how="semi")
        print(f"  src={src}: {sub_hit.height}/{sub.height} = {sub_hit.height / sub.height:.4%}", flush=True)

    missed = lab.join(cand, on=["q", "pid"], how="anti")
    print(f"missed true pairs: {missed.height}", flush=True)

    # Distribution of missed cases: did token blocking's top-K cutoff drop this pair (some token
    # overlap exists, just not enough to rank in the top-K), or is there no token overlap at all
    # (only a semantic/dense method could ever find it)? This is what tells us whether raising
    # block.py's k would already fix the gap vs. whether dense blocking is actually necessary.
    if missed.height > 0:
        pq = P["parquet"] / "train"
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr"]).sort("rid")
        s2 = pl.read_parquet(pq / "source2.parquet", columns=["rid", "core1", "addr"])
        s3 = pl.read_parquet(pq / "source3.parquet", columns=["rid", "core1", "addr"])
        pool = pl.concat([
            s2.with_columns((pl.col("rid").cast(pl.Int64) + 2 * PID_BASE).alias("pid")),
            s3.with_columns((pl.col("rid").cast(pl.Int64) + 3 * PID_BASE).alias("pid"))]).select("pid", "core1", "addr")
        m = missed.join(s1.rename({"rid": "q", "core1": "a_core1", "addr": "a_addr"}), on="q", how="left") \
                   .join(pool.rename({"core1": "b_core1", "addr": "b_addr"}), on="pid", how="left")
        m = m.with_columns(
            (pl.col("a_core1") + " " + pl.col("a_addr")).str.to_lowercase().str.split(" ").alias("a_tok"),
            (pl.col("b_core1") + " " + pl.col("b_addr")).str.to_lowercase().str.split(" ").alias("b_tok"),
        ).with_columns(
            pl.col("a_tok").list.set_intersection(pl.col("b_tok")).list.len().alias("shared")
        )
        zero_overlap = int((m["shared"] == 0).sum())
        truncated = missed.height - zero_overlap
        print(f"  missed breakdown (approximate, raw word overlap not block.py's exact tokenizer): "
              f"{truncated} likely top-K truncation (some shared tokens), "
              f"{zero_overlap} zero token overlap (no shared tokens -- dense retrieval's actual target)", flush=True)
    else:
        zero_overlap = truncated = 0

    log_stage("audit_recall", {}, {"recall": recall, "missed": float(missed.height),
                                    "missed_zero_overlap": float(zero_overlap), "seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
