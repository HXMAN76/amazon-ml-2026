"""F0.5 by how ambiguous an S1's candidate list is, and what France's ambiguity implies. Usage: python src/scripts/ambiguity_strata.py MODEL
Ambiguity of an S1 = number of its shortlisted pairs with model probability in [0.1, 0.9) (n_unc) and their summed probability. Both are
computed the same way on the labelled holdout and on the test set, so the holdout F0.5 per bucket can be reweighted to any test country.
Also sweeps the decision threshold inside the ambiguous holdout S1 (labelled): how much better a threshold tuned for them is than the global one."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q


def s1_stats(pp: pl.DataFrame) -> pl.DataFrame:
    return pp.group_by("q").agg(((pl.col("p") >= 0.1) & (pl.col("p") < 0.9)).sum().alias("n_unc"), (pl.col("p") * ((pl.col("p") >= 0.1) & (pl.col("p") < 0.9))).sum().alias("m_unc"),
                                pl.col("p").sum().alias("m_all"))


def bk(c: str) -> pl.Expr:
    return pl.col(c).cut([0, 1, 2, 3], labels=["0", "1", "2", "3", "4+"], left_closed=False).cast(pl.String).alias("b")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hq = holdout_q().astype(np.int64)
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": hq}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    st = s1_stats(ph)
    per = decision.per_entity_f05(decision.assign_exclusive(ph).filter(pl.col("p") >= thr), nt).join(st, on="q", how="left").with_columns(pl.col("n_unc").fill_null(0), pl.col("m_unc").fill_null(0.0))
    per = per.with_columns(bk("n_unc"))
    g = per.group_by("b").agg(pl.len().alias("s1"), (pl.len() / per.height).alias("share"), pl.col("f").mean().alias("f05"), ((1 - pl.col("f")).sum() / per.height).alias("loss_contrib")).sort("b")
    print(f"holdout F0.5 {float(per['f'].mean()):.5f}\n\nby number of uncertain pairs (p in [0.1,0.9)) of the S1\n{g}")

    te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    pt = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64))
    stt = te.join(s1_stats(pt), on="q", how="left").with_columns(pl.col("n_unc").fill_null(0)).with_columns(bk("n_unc"))
    sh = stt.group_by("ctry", "b").len().with_columns((pl.col("len") / pl.col("len").sum().over("ctry")).alias("share")).sort("ctry", "b")
    print(f"\ntest share of S1 by bucket\n{sh.pivot(on='b', index='ctry', values='share')}")
    fb = {r["b"]: r["f05"] for r in g.to_dicts()}
    for c in ("us", "india", "france"):
        s = sh.filter(pl.col("ctry") == c)
        print(f"reweighted F0.5 estimate {c}: {sum(r['share'] * fb.get(r['b'], 0.0) for r in s.to_dicts()):.5f}")

    own = decision.assign_exclusive(ph)
    print("\nthreshold sweep inside holdout S1 with n_unc >= 1, >= 2 (labelled)")
    for lo in (1, 2):
        sub = per.filter(pl.col("n_unc") >= lo)["q"]
        sub_nt = nt.join(pl.DataFrame({"q": sub}), on="q", how="semi")
        res = []
        for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
            f = decision.per_entity_f05(own.filter(pl.col("p") >= th), sub_nt)["f"].mean()
            res.append((th, round(float(f), 5)))
        print(f"n_unc >= {lo}: {sub.len()} S1 ({sub.len() / per.height:.4f} of the holdout); F0.5 by threshold {res}")


if __name__ == "__main__":
    main()
