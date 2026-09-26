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


def final_q(n: int = 100_000, seed: int = 4242) -> np.ndarray:
    """The second, untouched holdout (v6): n train S1 outside the sample, the locked holdout and the S1 the first two dense encoders
    were fine-tuned on. No model trains on them and no choice is made on them; they are scored once, at the freeze
    (src/scripts/final_check.py), to confirm the pick made on the locked holdout after many versions were compared there."""
    from ber.stages.dense import dense_train_q
    from ber.stages.dense_all import dense_all_train_q

    P, cfg = config.paths(), config.load()
    sample_q = pl.read_parquet(P["sample"] / "train_s1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"])["rid"].to_numpy().astype(np.int64)
    used = [sample_q, holdout_q().astype(np.int64), dense_train_q(cfg["dense"]).astype(np.int64), dense_all_train_q(cfg["dense_all"]).astype(np.int64)]
    free = np.setdiff1d(s1, np.concatenate(used))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(free, size=min(n, len(free)), replace=False))


def unscored_q() -> np.ndarray:
    """S1 no model may train on: the locked holdout and the final holdout."""
    return np.union1d(holdout_q().astype(np.int64), final_q())

