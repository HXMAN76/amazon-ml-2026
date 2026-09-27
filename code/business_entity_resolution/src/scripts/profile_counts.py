"""Does France's output look like the data generator? Usage: python src/scripts/profile_counts.py VARIANT MODEL
The number of true matches per S1 follows one distribution in train (labels): share of S1 with 0, 1, 2, ... matches, and with 0 to 5 S2 and 0 to 6 S3.
Compares that truth with the predicted counts per S1 of VARIANT (exclusive assignment at MODEL's threshold) for France, the US and India on the test,
and with the holdout's predictions (where the truth is known), to show where France's lists are too short or too long."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def dist(counts: pl.DataFrame, col: str, top: int) -> dict:
    c = counts[col].clip(0, top).to_numpy()
    return {i: round(float((c == i).mean()), 4) for i in range(top + 1)}


def main() -> None:
    var, model = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    hthr = json.loads((P["work"] / "models" / model / "holdout.json").read_text())["stack_threshold"]
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(pl.col("s1_rid").cast(pl.Int64).alias("q"))
    s1tr = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid"]).select(pl.col("rid").cast(pl.Int64).alias("q"))
    truth = s1tr.join(lab.group_by("q").agg(pl.len().alias("n"), (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")), on="q", how="left").fill_null(0)
    rows = [("train truth", dist(truth, "n", 8), dist(truth, "n2", 5), dist(truth, "n3", 6))]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    ph = pl.read_parquet(P["work"] / "models" / model / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    oh = decision.assign_exclusive(ph).filter(pl.col("p") >= hthr).with_columns((pl.col("pid") // PID_BASE).alias("src"))
    ch = hq.join(oh.group_by("q").agg(pl.len().alias("n"), (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")), on="q", how="left").fill_null(0)
    th = hq.join(truth, on="q", how="left").fill_null(0)
    rows += [("holdout truth", dist(th, "n", 8), dist(th, "n2", 5), dist(th, "n3", 6)), ("holdout predicted", dist(ch, "n", 8), dist(ch, "n2", 5), dist(ch, "n3", 6))]
    s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "ctry")
    pp = pl.read_parquet(P["work"] / "output" / var / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    ot = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).with_columns((pl.col("pid") // PID_BASE).alias("src"))
    ct = s1.join(ot.group_by("q").agg(pl.len().alias("n"), (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")), on="q", how="left").fill_null(0)
    for c in ("us", "india", "france"):
        x = ct.filter(pl.col("ctry") == c)
        rows.append((f"test {c} predicted", dist(x, "n", 8), dist(x, "n2", 5), dist(x, "n3", 6)))
    print(f"share of S1 by number of matches (8 = 8 or more), S2 matches, S3 matches; {var} at threshold {thr:.2f}")
    for name, n, n2, n3 in rows:
        print(f"\n{name}\n  n : {n}\n  n2: {n2}\n  n3: {n3}")


if __name__ == "__main__":
    main()
