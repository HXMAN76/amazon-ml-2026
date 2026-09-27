"""What would the France swap rules score if France behaved like the labelled countries? Applies decode_rules' swap rules to the holdout.
Usage: python src/scripts/rule_on_holdout.py MODEL. Prints the holdout macro F0.5 with no rule, with `swap_exact` and with `swap` (pmax 0.9999)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hq = holdout_q().astype(np.int64)
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": hq}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(ph).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    own = flag(own, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    exact = own.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q").len().rename({"len": "n_exact"})
    own = own.join(exact, on="q", how="left").with_columns(pl.col("n_exact").fill_null(0))
    base = decision.macro_f05(own.select("q", "pid", "p", "label"), nt)
    print(f"model {name}, threshold {thr:.2f}: holdout macro F0.5 with no rule {base:.5f}")
    for rule, cond in (("swap_exact", pl.col("swap") & (pl.col("p") < 0.9999) & (pl.col("n_exact") >= 1)), ("swap", pl.col("swap") & (pl.col("p") < 0.9999))):
        dropped = own.filter(cond)
        kept = own.filter(~cond)
        f = decision.macro_f05(kept.select("q", "pid", "p", "label"), nt)
        print(f"rule {rule}: drops {dropped.height} of {own.height} predicted pairs ({dropped.height / own.height:.4f}), true share of the dropped {float(dropped['label'].mean()):.4f}; holdout macro F0.5 {f:.5f} ({f - base:+.5f})")


if __name__ == "__main__":
    main()
