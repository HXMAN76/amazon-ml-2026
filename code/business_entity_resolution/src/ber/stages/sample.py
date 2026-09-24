"""sample: choose the training queries (S1 rows) and assign CV folds.

Query sampling with the full S2/S3 pool kept (see plan.md section 2): only S1 is subsampled, so
distractor density and competition stay the same as at test time.
Output: WORK/sample/train_s1.parquet with rid, ctry, n_matches, fold.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ber import config
from ber.tracking import log_stage


def main() -> None:
    P, prm = config.paths(), config.load()["sample"]
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"])
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet")
    cnt = lab.group_by("s1_rid").len().rename({"s1_rid": "rid", "len": "n_matches"})
    s1 = s1.join(cnt, on="rid", how="left").with_columns(pl.col("n_matches").fill_null(0).cast(pl.Int32))
    rng = np.random.default_rng(prm["seed"])
    n = min(prm["n_s1"], s1.height)
    idx = np.sort(rng.choice(s1.height, size=n, replace=False))
    smp = s1[idx].with_columns(pl.Series("fold", rng.integers(0, prm["folds"], size=n), dtype=pl.Int8))
    out = P["sample"] / "train_s1.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    smp.write_parquet(out, compression="zstd")
    mix = smp["ctry"].value_counts()
    print(f"sampled {n} of {s1.height} S1; country mix: {dict(zip(mix['ctry'], mix['count']))}; "
          f"singleton rate {(smp['n_matches'] == 0).mean():.4f} (population {(s1['n_matches'] == 0).mean():.4f})", flush=True)
    log_stage("sample", prm, {"n_s1": n, "singleton_rate": float((smp["n_matches"] == 0).mean()),
                              "mean_matches": float(smp["n_matches"].mean())})


if __name__ == "__main__":
    main()
