"""Twin S1: other S1 at the same exact address with a near-identical name. Share per country and the holdout F0.5 of such S1.
Usage: python src/scripts/twin_strata.py MODEL
Twin level of an S1 = highest token Jaccard of its core name against another S1 at the same normalised address (0 when it is alone at its
address); buckets: alone, 0-0.5, 0.5-0.99, identical core name."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q


def twin_level(s1: pl.DataFrame, only: np.ndarray | None = None) -> pl.DataFrame:
    """(q, twin) for every S1 (or the ones in `only`)."""
    d = s1.select("q", "addr", "core1").filter(pl.col("addr").str.len_chars() > 0)
    g = d.group_by("addr").len().filter(pl.col("len") >= 2).select("addr")
    d = d.join(g, on="addr", how="semi").with_columns(pl.col("core1").str.split(" ").list.unique().alias("tok"))
    a = d.rename({"q": "q1", "core1": "c1", "tok": "t1"})
    if only is not None:
        a = a.join(pl.DataFrame({"q1": only}), on="q1", how="semi")
    pairs = a.join(d.rename({"q": "q2", "core1": "c2", "tok": "t2"}), on="addr").filter(pl.col("q1") != pl.col("q2"))
    inter = pairs.select(pl.col("t1").list.set_intersection(pl.col("t2")).list.len().alias("i"), pl.col("t1").list.len().alias("n1"), pl.col("t2").list.len().alias("n2"), "q1", "c1", "c2")
    j = inter.with_columns(pl.when(pl.col("c1") == pl.col("c2")).then(1.0).otherwise(pl.col("i") / (pl.col("n1") + pl.col("n2") - pl.col("i"))).alias("jac"))
    return j.group_by("q1").agg(pl.col("jac").max().alias("twin")).rename({"q1": "q"})


def bucket(t: pl.Expr) -> pl.Expr:
    return (pl.when(t.is_null()).then(pl.lit("0 alone at its address")).when(t < 0.5).then(pl.lit("1 shares address, names differ (<0.5)"))
              .when(t < 0.99).then(pl.lit("2 shares address, similar names (0.5-0.99)")).otherwise(pl.lit("3 shares address, identical core name")))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    tr = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    hq = holdout_q().astype(np.int64)
    ttr = twin_level(tr, hq)
    tte = twin_level(te)
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": hq}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    pred = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    f = decision.per_entity_f05(decision.assign_exclusive(pred).filter(pl.col("p") >= thr), nt).join(ttr, on="q", how="left").join(tr.select("q", "ctry"), on="q", how="left")
    f = f.with_columns(bucket(pl.col("twin")).alias("b"))
    print(f"holdout F0.5 {float(f['f'].mean()):.5f}")
    g = f.group_by("b").agg(pl.len().alias("s1"), (pl.len() / f.height).alias("share"), pl.col("f").mean().alias("f05")).sort("b")
    print("\nholdout by twin bucket\n", g)
    t = te.join(tte, on="q", how="left").with_columns(bucket(pl.col("twin")).alias("b"))
    sh = t.group_by("ctry", "b").len().with_columns((pl.col("len") / pl.col("len").sum().over("ctry")).alias("share")).sort("ctry", "b")
    print("\ntest share by country and twin bucket\n", sh)
    fb = {r["b"]: r["f05"] for r in g.to_dicts()}
    for c in ("us", "india", "france"):
        s = sh.filter(pl.col("ctry") == c)
        est = sum(r["share"] * fb.get(r["b"], float(f["f"].mean())) for r in s.to_dicts())
        print(f"estimated {c} F0.5 from the holdout twin-bucket scores: {est:.5f}")


if __name__ == "__main__":
    main()
