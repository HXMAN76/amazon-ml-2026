"""Do the cross-encoder and the first stage agree with the final decision, per country? Usage: python src/scripts/agreement.py MODEL XENC_DIR
Among the pairs the final model predicts (exclusive assignment, threshold): the share whose cross-encoder score xs (from XENC_DIR/{split}_xs.parquet,
scored on every shortlisted pair) is below 0.5 / 0.2 / 0.05, and whose first-stage probability p1 is below 0.5. On the labelled holdout the true share
of the pairs the cross-encoder doubts tells how much those doubts are worth. A country where the cross-encoder doubts many more of the final
predictions than in the holdout is a country where the stack trusts features that the text model does not."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from ber.stages.stack import load_p1


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    xdir = sys.argv[2] if len(sys.argv) > 2 else "xenc2F_v7"
    base = sys.argv[3] if len(sys.argv) > 3 else "v7"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    x = pl.read_parquet(P["work"] / xdir / "test_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    p1 = load_p1("test", base).select("q", "pid", pl.col("p").alias("p1"))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    sel = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(x, on=["q", "pid"], how="left").join(p1, on=["q", "pid"], how="left").join(te, on="q", how="left")

    def rep(d: pl.DataFrame, title: str, lab: bool) -> None:
        n = d.height
        cols = [(pl.col("xs") < 0.5).mean().alias("xs<0.5"), (pl.col("xs") < 0.2).mean().alias("xs<0.2"), (pl.col("xs") < 0.05).mean().alias("xs<0.05"),
                (pl.col("p1") < 0.5).mean().alias("p1<0.5"), (pl.col("p1") < 0.9).mean().alias("p1<0.9"), pl.col("xs").is_null().mean().alias("xs_missing")]
        print(f"\n== {title}: {n} predicted pairs")
        print(d.select(cols))
        if lab:
            for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)):
                s = d.filter((pl.col("xs") >= lo) & (pl.col("xs") < hi))
                print(f"   xs in [{lo},{hi}): {s.height} pairs, true share {float(s['label'].mean()) if s.height else float('nan'):.4f}")

    for c in ("france", "us", "india"):
        rep(sel.filter(pl.col("ctry") == c), f"test {c}", False)
    tr = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    xh = pl.read_parquet(P["work"] / xdir / "train_xs.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    p1h = load_p1("train", base).join(hq, on="q", how="semi").select("q", "pid", pl.col("p").alias("p1"))
    selh = decision.assign_exclusive(ph).filter(pl.col("p") >= thr).join(xh, on=["q", "pid"], how="left").join(p1h, on=["q", "pid"], how="left").join(tr, on="q", how="left")
    rep(selh, "holdout (US and India, labelled)", True)


if __name__ == "__main__":
    main()
