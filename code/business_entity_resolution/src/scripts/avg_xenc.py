"""Average two symmetric cross-encoder seeds into one score file (as used by stacks s27 and s28). Usage: python src/scripts/avg_xenc.py OUT DIR_A DIR_B
For each split, joins WORK/DIR_A/{split}_xs.parquet and WORK/DIR_B/{split}_xs.parquet (columns xs, xs_asym, scored with symmetric=1) and writes
WORK/OUT/{split}_xs.parquet with xs = sigmoid of the mean logit, xs_asym = mean of the two asymmetries, xs_seed_gap = |logit_A - logit_B|."""

import sys

import numpy as np
import polars as pl

from ber import config


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def main() -> None:
    out, da, db = sys.argv[1:4]
    W = config.paths()["work"]
    (W / out).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        a = pl.read_parquet(W / da / f"{split}_xs.parquet")
        b = pl.read_parquet(W / db / f"{split}_xs.parquet").select("q", "pid", pl.col("xs").alias("xs_b"), pl.col("xs_asym").alias("asym_b"))
        m = a.join(b, on=["q", "pid"], how="inner")
        la, lb = logit(m["xs"].to_numpy()), logit(m["xs_b"].to_numpy())
        r = m.select("q", "pid").with_columns(pl.Series("xs", (1 / (1 + np.exp(-(la + lb) / 2))).astype(np.float32)),
                                              ((m["xs_asym"] + m["asym_b"]) / 2).cast(pl.Float32).alias("xs_asym"),
                                              pl.Series("xs_seed_gap", np.abs(la - lb).astype(np.float32)))
        r.write_parquet(W / out / f"{split}_xs.parquet", compression="zstd")
        print(f"{split}: {a.height} and {b.height} pairs, {r.height} averaged", flush=True)


if __name__ == "__main__":
    main()
