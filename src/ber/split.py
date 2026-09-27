"""The locked holdout split, kept dependency-light so every stage (also the torch environment) can import it."""

from __future__ import annotations

import numpy as np
import polars as pl

from ber import config


def holdout_q(n: int = 150_000, seed: int = 2026) -> np.ndarray:
    """The locked holdout: n train S1 drawn once with a fixed seed from those outside the training sample."""
    P = config.paths()
    sample_q = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"]).rename({"rid": "q"})
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"]).rename({"rid": "q"})
    rest = s1.join(sample_q, on="q", how="anti")
    rng = np.random.default_rng(seed)
    return rest["q"].to_numpy()[np.sort(rng.choice(rest.height, size=min(n, rest.height), replace=False))]
