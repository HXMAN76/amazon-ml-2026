"""How small can the reported candidate set be without losing score?  Usage: python src/scripts/shortlist_eval.py STACK BASE

The final rank also rewards a smaller candidate set per S1. A shortlist keeps, per S1, the K candidates with the highest first-stage
probability p1 (and p1 >= pmin). Predictions of the stacked model STACK are restricted to the shortlist and the locked-holdout macro F0.5
is recomputed (threshold and exclusive assignment as in the model), next to the mean shortlist size and the share of true pairs kept.
"""

import json
import sys

import polars as pl

from ber import config, decision
from ber.split import holdout_q
from ber.stages.stack import load_p1


def main() -> None:
    stack, base = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / stack / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q()})
    pred = pl.read_parquet(P["work"] / "models" / stack / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    p1 = load_p1("train", base).join(hq, on="q", how="semi").select("q", "pid", pl.col("p").alias("p1")).with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").alias("r1"))
    d = pred.join(p1, on=["q", "pid"], how="left")
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    nt = hq.join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    n_true_pairs = nt["n_true"].sum()
    full = decision.macro_f05(decision.assign_exclusive(pred).filter(pl.col("p") >= thr), nt)
    print(f"{stack} on base {base}: full candidate set {pred.height / hq.height:.1f} per S1, true pairs in candidates {pred['label'].sum() / n_true_pairs:.4f} of all, macro F0.5 {full:.4f}")
    for k in (3, 4, 5, 6, 8, 10, 12, 15, 20):
        for pmin in (0.0, 0.005, 0.02):
            s = d.filter((pl.col("r1") <= k) & (pl.col("p1") >= pmin))
            f = decision.macro_f05(decision.assign_exclusive(s).filter(pl.col("p") >= thr), nt)
            print(f"K={k:2d} pmin={pmin:.3f}: {s.height / hq.height:5.1f} per S1, true pairs kept {s['label'].sum() / n_true_pairs:.4f}, "
                  f"macro F0.5 {f:.4f} ({f - full:+.4f})")


if __name__ == "__main__":
    main()
