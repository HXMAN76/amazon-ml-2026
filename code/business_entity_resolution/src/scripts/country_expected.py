"""Country-level expected F0.5 from the model's own probabilities (a France proxy: France has no labels).

Usage: python src/scripts/country_expected.py MODEL
For every S1 the estimate is F = 1.25 TP / (0.25 nTrue + nPred) with TP = sum of p over the selected pairs and nTrue = sum of p over the
pairs the S1 owns after exclusive assignment (its expected number of found matches); an S1 with no selected pair scores prod(1 - p).
It ignores blocking misses. The same estimator is run on the labelled holdout, where the actual score is known, to measure its bias per
country, and on the test set for US, India and France. Also reports the mean expected number of found matches per S1 (train truth 3.46).
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision


def estimate(pairs: pl.DataFrame, thr: float) -> pl.DataFrame:
    own = decision.assign_exclusive(pairs.select("q", "pid", "p"))
    g = own.group_by("q").agg(pl.col("p").sum().alias("sum_p"), (1 - pl.col("p")).log().sum().exp().alias("p_none"),
                              (pl.col("p") >= thr).sum().alias("n_pred"), (pl.col("p") * (pl.col("p") >= thr)).sum().alias("tp"))
    f = pl.when(pl.col("n_pred") > 0).then(1.25 * pl.col("tp") / (0.25 * pl.col("sum_p") + pl.col("n_pred"))).otherwise(pl.col("p_none"))
    return g.with_columns(f.alias("f_est"))


def main() -> None:
    name = sys.argv[1]
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    # labelled holdout: estimator versus actual, per country
    s1tr = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"})
    hp = pl.read_parquet(mdl / "holdout_pred.parquet")
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"})
    hq = hp.select("q").unique().join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    actual = decision.per_entity_f05(decision.assign_exclusive(hp).filter(pl.col("p") >= thr), hq).rename({"f": "f_act"})
    est = estimate(hp, thr)
    h = hq.join(actual, on="q").join(est, on="q", how="left").join(s1tr, on="q").with_columns(pl.col("f_est").fill_null(1.0), pl.col("sum_p").fill_null(0.0))
    print(f"== {name}: holdout, actual versus estimated macro F0.5 (threshold {thr:.2f})")
    for (c,), g in h.group_by("ctry"):
        print(f"{c:8s} n={g.height:7d} actual {g['f_act'].mean():.4f} estimated {g['f_est'].mean():.4f} bias(est-act) {g['f_est'].mean() - g['f_act'].mean():+.4f} "
              f"| mean found matches est {g['sum_p'].mean():.2f} true {g['n_true'].mean():.2f}")
    bias = {c: float(g["f_est"].mean() - g["f_act"].mean()) for (c,), g in h.group_by("ctry")}
    # test set
    s1te = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).rename({"rid": "q"})
    tp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet", columns=["q", "pid", "p"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    t = s1te.join(estimate(tp, thr), on="q", how="left").with_columns(pl.col("f_est").fill_null(1.0), pl.col("sum_p").fill_null(0.0), pl.col("n_pred").fill_null(0))
    print(f"== {name}: test set, estimated (blocking misses ignored)")
    share = {}
    for (c,), g in t.group_by("ctry"):
        share[c] = g.height
        print(f"{c:8s} n={g.height:8d} estimated {g['f_est'].mean():.4f} mean found matches {g['sum_p'].mean():.2f} mean predicted {g['n_pred'].mean():.2f} "
              f"uncertain-lead share {float(((g['sum_p'] > 0.2) & (g['n_pred'] == 0)).mean()):.3f}")
    n = sum(share.values())
    fr = t.filter(pl.col("ctry") == "france")["f_est"].mean()
    print(f"test-mix estimate {sum(t.filter(pl.col('ctry') == c)['f_est'].mean() * k for c, k in share.items()) / n:.4f}; "
          f"France est {fr:.4f} vs US est {t.filter(pl.col('ctry') == 'us')['f_est'].mean():.4f} (US holdout bias {bias.get('us', float('nan')):+.4f})")


if __name__ == "__main__":
    main()
