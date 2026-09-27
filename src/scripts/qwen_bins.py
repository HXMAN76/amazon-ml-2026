"""Where do the confident look-alikes live, and how much of each first-stage probability bin is already scored by Qwen? Usage: python src/scripts/qwen_bins.py
All shortlisted pairs (WORK/xenc2F_v7 lists, first-stage p1 of v7) by p1 bin: pairs in train and test, share already scored by Qwen (WORK/xenc3Q_v7 band scores), and on the labelled train
pairs the number of false pairs (look-alikes) and true pairs, i.e. what an extra cross-encoder score could still fix outside the band."""

import numpy as np
import polars as pl

from ber import config

EDGES = [0.0, 0.005, 0.02, 0.1, 0.5, 0.9, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0001]


def main() -> None:
    W = config.paths()["work"]
    out = {}
    for split in ("train", "test"):
        allp = pl.read_parquet(W / "xenc2F_v7" / f"{split}.parquet", columns=["q", "pid", "p"] + (["label"] if split == "train" else []))
        band = pl.read_parquet(W / "xenc3Q_v7" / f"{split}_xs.parquet").select("q", "pid").with_columns(pl.lit(True).alias("qwen"))
        d = allp.join(band, on=["q", "pid"], how="left").with_columns(pl.col("qwen").fill_null(False))
        d = d.with_columns(pl.col("p").cut(EDGES[1:-1], left_closed=True).alias("bin"))
        agg = [pl.len().alias("pairs"), pl.col("qwen").mean().alias("qwen_share")]
        if split == "train":
            agg += [(pl.col("label") == 0).sum().alias("false"), (pl.col("label") == 1).sum().alias("true")]
        out[split] = d.group_by("bin").agg(agg).sort("bin")
        print(f"\n== {split}: {allp.height} shortlisted pairs, {int(d['qwen'].sum())} already scored by Qwen")
        with pl.Config(tbl_rows=30, tbl_width_chars=180):
            print(out[split])
    tr, te = out["train"], out["test"]
    j = tr.join(te, on="bin", suffix="_test").with_columns((pl.col("pairs") * (1 - pl.col("qwen_share"))).alias("train_unscored"), (pl.col("pairs_test") * (1 - pl.col("qwen_share_test"))).alias("test_unscored"))
    j = j.with_columns(((pl.col("train_unscored") + pl.col("test_unscored")) / 1e6).alias("M_pairs_to_score"), (pl.col("false") * (1 - pl.col("qwen_share"))).alias("false_unscored"))
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print("\n== pairs still to score per bin (train + test, millions) against the false pairs they hold (train, unscored)")
        print(j.select("bin", "M_pairs_to_score", "false_unscored", "true").sort("bin"))


if __name__ == "__main__":
    main()
