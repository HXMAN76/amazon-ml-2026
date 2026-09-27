"""Recall and precision by noise type on the locked holdout.  Usage: python src/scripts/noise_analysis.py STACK_MODEL

Every true holdout pair is classified by how the pool record differs from the S1 record (first matching rule wins): non-Latin name,
alias (names share almost nothing), glued name (equal once spaces are removed), suffix or prefix words injected or dropped (one token
set contains the other), typo (same word count, similar words), otherwise exact or near-exact. Independently, the house numbers are
compared (equal, one digit dropped or added, other change, missing) and the pool address is checked for being empty.
Reported per category: true pairs, share in the candidate set, share predicted by the model (recall), and, for the pairs the model
predicted that are false, the same classification (false positives by type).
"""

import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def classify(a: str, b: str, nl: float) -> str:
    if nl > 0.5:
        return "1 non-Latin name"
    ta, tb = set(a.split()), set(b.split())
    if fuzz.token_set_ratio(a, b) < 40:
        return "2 alias (names share almost nothing)"
    if a.replace(" ", "") == b.replace(" ", "") and a != b:
        return "3 glued or split name"
    if a == b:
        return "7 identical name"
    if ta and tb and (ta < tb or tb < ta):
        return "4 words injected or dropped"
    if len(ta) == len(tb) and fuzz.token_sort_ratio(a, b) >= 70:
        return "5 typos"
    return "6 other name change"


def house(a: str, b: str) -> str:
    import re

    ha = re.match(r"\s*(\d+)", a or ""); hb = re.match(r"\s*(\d+)", b or "")
    if not b:
        return "pool address empty"
    if not ha or not hb:
        return "house number missing"
    x, y = ha.group(1).lstrip("0"), hb.group(1).lstrip("0")
    if x == y:
        return "house number equal"
    if x.startswith(y) or y.startswith(x):
        return "one digit dropped or added"
    return "house number differs"


def main() -> None:
    name = sys.argv[1]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q()})
    pred = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    chosen = decision.assign_exclusive(pred).filter(pl.col("p") >= thr).select("q", "pid").with_columns(pl.lit(1).alias("pred"))
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q")).select("q", "pid").join(hq, on="q", how="semi")
    cols = ["rid", "core1", "addr", "nl_name"]
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1", "addr"]).rename({"rid": "q", "core1": "an", "addr": "aa"})
    pool = pl.concat([pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=cols).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"))
                      for s in (2, 3)]).drop("rid").rename({"core1": "bn", "addr": "ba"})

    def describe(df: pl.DataFrame) -> pl.DataFrame:
        d = df.join(s1, on="q", how="left").join(pool, on="pid", how="left")
        cat = [classify(x or "", y or "", n or 0.0) for x, y, n in zip(d["an"].to_list(), d["bn"].to_list(), d["nl_name"].to_list())]
        hs = [house(x, y) for x, y in zip(d["aa"].to_list(), d["ba"].to_list())]
        return d.with_columns(pl.Series("cat", cat), pl.Series("house", hs))

    t = describe(lab.join(pred.select("q", "pid").with_columns(pl.lit(1).alias("cand")), on=["q", "pid"], how="left").join(chosen, on=["q", "pid"], how="left")
                 .with_columns(pl.col("cand").fill_null(0), pl.col("pred").fill_null(0)))
    print(f"true holdout pairs {t.height}; overall candidate share {t['cand'].mean():.4f}, predicted share (recall) {t['pred'].mean():.4f}")
    for key in ("cat", "house"):
        g = t.group_by(key).agg(pl.len().alias("pairs"), pl.col("cand").mean().alias("in_candidates"), pl.col("pred").mean().alias("recall")).sort(key)
        g = g.with_columns((pl.col("pairs") * (1 - pl.col("recall"))).round(0).alias("missed_pairs"), (pl.col("pairs") / t.height).round(4).alias("share"))
        print(f"\n== true pairs by {key}")
        print(g)
    truth = lab.with_columns(pl.lit(1).alias("true"))
    fp = chosen.join(truth, on=["q", "pid"], how="anti")
    f = describe(fp)
    print(f"\n== false positives ({fp.height} predicted pairs are false) by name category and house number")
    print(f.group_by("cat").len().sort("cat"))
    print(f.group_by("house").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
