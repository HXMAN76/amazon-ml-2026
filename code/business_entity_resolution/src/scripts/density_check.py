"""What are the extra ownerless pool records of the test set: look-alike decoys or orphans, and how much thinning would they justify?

Usage: python src/scripts/density_check.py MODEL        (MODEL: a stacked model with output/MODEL/pair_p.parquet, e.g. bs_s6)

Test has about 5.8 pool records per S1 against 4.7 in train, with the same number of matches per S1 (3.46 predicted and in train), so
it has about twice as many ownerless records per S1. Two kinds are possible and they need different training:
  * look-alike decoys of an existing S1 (train has these): their nearest S1 in the dense_all embedding is very close;
  * orphans, i.e. copies of a business whose S1 is absent from the test file: their nearest S1 is only some other business.
Dropping train S1 (thinning) would create exactly the second kind. Result on 26 Sep: orphan share 0, the extra test
records are decoys, so thinning was dropped. Label-free on test, this script compares the nearest-S1 cosine of the test records
the model leaves unclaimed with two labelled train references (ownerless train records = decoys; true records with their owner
removed = simulated orphans), fits the test histogram as a mixture of the two, and turns the orphan share into a thinning fraction:
dropping a fraction f of train S1 gives m * f / (1 - f) orphans per remaining S1 (m = true matches per S1), so f = o / (m + o) for
o orphans per test S1. Reads only test inputs and the model's own test predictions; nothing is trained on test.
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.stages.block import PID_BASE

BINS = np.linspace(0.0, 1.0, 51)


def hist(x: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(np.clip(x, 0, 1), bins=BINS)
    return h / max(h.sum(), 1)


def mixture_weight(target: np.ndarray, orphan: np.ndarray, decoy: np.ndarray) -> float:
    """w in [0, 1] minimising the L1 distance between hist(target) and w hist(orphan) + (1 - w) hist(decoy)."""
    ht, ho, hd = hist(target), hist(orphan), hist(decoy)
    grid = np.linspace(0, 1, 101)
    return float(grid[np.argmin([np.abs(ht - (w * ho + (1 - w) * hd)).sum() for w in grid])])


def main() -> None:
    name = sys.argv[1]
    P = config.paths()
    n = {}
    for split in ("train", "test"):
        pq = P["parquet"] / split
        n[split] = {s: pl.scan_parquet(pq / f"source{s}.parquet").select(pl.len()).collect().item() for s in (1, 2, 3)}
    pool_per_s1 = {k: (v[2] + v[3]) / v[1] for k, v in n.items()}
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").select(
        pl.col("s1_rid").cast(pl.Int64).alias("owner"), (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"))
    m = lab.height / n["train"][1]

    tr = pl.read_parquet(P["work"] / "dense_all" / "train" / "pairs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    decoy = tr.filter(pl.col("rank") == 0).join(lab, on="pid", how="anti")["cos"].to_numpy()
    orphan = (tr.join(lab, on="pid", how="inner").filter(pl.col("q") != pl.col("owner"))
                .group_by("pid").agg(pl.col("cos").max())["cos"].to_numpy())  # best S1 once the owner is gone (within the top 5)

    te = pl.read_parquet(P["work"] / "dense_all" / "test" / "pairs.parquet").with_columns(pl.col("pid").cast(pl.Int64))
    cfg = json.loads((P["work"] / "models" / name / "config.json").read_text())
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("pid").cast(pl.Int64))
    claimed = decision.assign_exclusive(pp).filter(pl.col("p") >= cfg["threshold"]).select("pid").unique()
    unclaimed = te.filter(pl.col("rank") == 0).join(claimed, on="pid", how="anti")["cos"].to_numpy()

    w = mixture_weight(unclaimed, orphan, decoy)
    unclaimed_per_s1 = len(unclaimed) / n["test"][1]
    o = w * unclaimed_per_s1
    rep = {
        "pool_per_s1": pool_per_s1, "f_from_pool_sizes": 1 - pool_per_s1["train"] / pool_per_s1["test"], "matches_per_s1_train": m,
        "train_ownerless_per_s1": (n["train"][2] + n["train"][3] - lab.height) / n["train"][1],
        "test_unclaimed_per_s1": unclaimed_per_s1, "claimed_per_s1_test": claimed.height / n["test"][1],
        "mean_top1_cos": {"train_decoys": float(decoy.mean()), "train_simulated_orphans": float(orphan.mean()),
                          "test_unclaimed": float(unclaimed.mean())},
        "orphan_share_of_test_unclaimed": w, "orphans_per_test_s1": o, "thin_frac": o / (m + o),
    }
    out = P["work"] / "measure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "density.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"DENSITY: orphan share {w:.2f} of the unclaimed test records -> thin frac {rep['thin_frac']:.3f} "
          f"(pool-size bound {rep['f_from_pool_sizes']:.3f})", flush=True)


if __name__ == "__main__":
    main()
