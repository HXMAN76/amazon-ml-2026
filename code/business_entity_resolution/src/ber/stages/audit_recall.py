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
    print(f"missed true pairs: {missed.height} ({time.time() - t0:.0f}s)", flush=True)
    log_stage("audit_recall", {}, {"recall": recall, "missed": float(missed.height), "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
