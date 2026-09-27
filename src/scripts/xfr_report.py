"""Is a France-aware cross-encoder separating France's decoys from its true copies? Usage: python src/scripts/xfr_report.py MODEL XDIR [OLD_XDIR]
Label-free on France (test): for the predicted pairs of MODEL (exclusive assignment, threshold), by kind of name relation, the mean score and the
share scored below 0.5 by the new cross-encoder (WORK/XDIR/test_xs.parquet) against the old one (OLD_XDIR, default xenc2F_v7). Wanted: low scores on
type-word swaps and on sure negatives (pairs beyond an S1's 5 S2 / 6 S3 exact copies, which cannot be true), high scores on exact names, initials,
spelled legal forms and swaps into France's noise words (fils, groupe, services, developpement; research.md 25). Labelled on the locked holdout
(WORK/XDIR/train_xs.parquet): average precision of new against old, and the share of the model's true and false predicted pairs below 0.5."""

import json
import sys

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from ber import config, decision
from ber.split import holdout_q
from word_swap import flag, strip_spaced, tok_df

PID_BASE = 10_000_000
TYPE = {"club", "ecole", "comite", "amicale", "sportive", "amis", "parents", "union", "college", "primaire", "fetes", "compagnie", "centre"}
NOISE_FR = {"fils", "groupe", "services", "developpement"}


def main() -> None:
    name, xdir = sys.argv[1], sys.argv[2]
    old = sys.argv[3] if len(sys.argv) > 3 else "xenc2F_v7"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src"))
                      for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    d = d.with_columns(pl.col("b_core").str.split(" ").list.set_difference(pl.col("a_core").str.split(" ")).list.first().alias("xb"))

    def xs_of(dirname: str, alias: str) -> pl.DataFrame:
        return pl.read_parquet(P["work"] / dirname / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias(alias))

    d = d.join(xs_of(xdir, "x_new"), on=["q", "pid"], how="left").join(xs_of(old, "x_old"), on=["q", "pid"], how="left")
    exact = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    d = d.join(exact, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    tiny = (pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ")
    ini = d.filter(tiny).select("q", "pid", "a_core", "b_core").rows()
    ini_ok = {(q, pid) for q, pid, ac, bc in ini if (lambda it: all(c in it for c in bc))(iter("".join(w[0] for w in ac.split())))}
    d = d.with_columns(pl.Series("ini", [(q, pid) in ini_ok for q, pid in zip(d["q"].to_list(), d["pid"].to_list())]))
    kind = (pl.when(pl.col("core_eq")).then(pl.lit("exact"))
            .when(~pl.col("core_eq") & (pl.col("k") >= pl.when(pl.col("src") == 2).then(5).otherwise(6))).then(pl.lit("sure_negative"))
            .when(pl.col("swap") & pl.col("xb").is_in(list(TYPE))).then(pl.lit("swap_type"))
            .when(pl.col("swap") & pl.col("xb").is_in(list(NOISE_FR))).then(pl.lit("swap_noise"))
            .when(pl.col("swap")).then(pl.lit("swap_other"))
            .when(pl.col("ini")).then(pl.lit("initials"))
            .when(strip_spaced(pl.col("a_core")) == strip_spaced(pl.col("b_core"))).then(pl.lit("spelled_legal"))
            .otherwise(pl.lit("other")))
    r = (d.with_columns(kind.alias("kind")).group_by("kind")
          .agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), pl.col("x_old").mean().alias("old_mean"), pl.col("x_new").mean().alias("new_mean"),
               (pl.col("x_old") < 0.5).mean().alias("old_below_0.5"), (pl.col("x_new") < 0.5).mean().alias("new_below_0.5"),
               (pl.col("x_new") < 0.1).mean().alias("new_below_0.1"), pl.col("x_new").is_null().mean().alias("unscored")).sort("pairs", descending=True))
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220):
        print(f"France predicted pairs of {name} ({d.height}) by kind: old {old} against new {xdir}\n{r}")
    h = pl.read_parquet(P["work"] / xdir / "train_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("x_new"))
    ho = (pl.read_parquet(P["work"] / old / "train_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("x_old"))
            .join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi"))
    hp = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    hthr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    hd = hp.join(h, on=["q", "pid"], how="inner").join(ho, on=["q", "pid"], how="inner")
    y = hd["label"].to_numpy()
    print(f"holdout pairs scored by both: {hd.height}; average precision new {average_precision_score(y, hd['x_new'].to_numpy()):.5f} "
          f"old {average_precision_score(y, hd['x_old'].to_numpy()):.5f}")
    pr = decision.assign_exclusive(hd).filter(pl.col("p") >= hthr)
    for lab, nm in ((1, "true"), (0, "false")):
        g = pr.filter(pl.col("label") == lab)
        print(f"holdout predicted {nm} pairs {g.height}: below 0.5 new {(g['x_new'] < 0.5).mean():.5f} old {(g['x_old'] < 0.5).mean():.5f}; "
              f"below 0.1 new {(g['x_new'] < 0.1).mean():.5f} old {(g['x_old'] < 0.1).mean():.5f}")


if __name__ == "__main__":
    main()
