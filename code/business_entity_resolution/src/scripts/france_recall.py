"""Where are France's missing true copies? Usage: python src/scripts/france_recall.py MODEL [XDIR]
The generator gives an S1 about 3.46 matches (train); the model and the France rules keep about 3.2 to 3.3 per France S1, and 12% of France's
unassigned pool records have a best probability of 0.1 to 0.72 (US 3%). This looks at the shortlisted pairs whose pool record nobody owns (a restore
candidate: adding it cannot take a record from another S1) by the kind of name relation a true France copy shows (exact core, equal after spaced
legal forms, initials of the S1's core, glued core, a swap into France's noise words) and by the address evidence (same house number, address
similarity >= 90). For each cell: pairs, probability, the France-aware cross-encoder score if XDIR is given, and the S1's free slots. The same
cells on the labelled holdout (US and India) give the true share of such restores where labels exist."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from word_swap import strip_spaced

PID_BASE = 10_000_000
NOISE_FR = ["fils", "groupe", "services", "developpement"]


def feats(P, split: str, keys: pl.DataFrame) -> pl.DataFrame:
    dirs = ["train", "train_rest"] if split == "train" else [split]
    files = [str(f) for d in dirs for f in sorted((P["work"] / "features" / d).glob("part_*.parquet"))]
    return (pl.scan_parquet(files).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset")
              .join(keys.lazy(), on=["q", "pid"], how="semi").collect())


def texts(P, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                                                                                                pl.col("core1").alias("b_core"), pl.lit(s).alias("src")) for s in (2, 3)])
    return s1, pool


def restore_candidates(pp: pl.DataFrame, thr: float, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Shortlisted pairs below the decision whose pool record no S1 owns, with the kind of name relation and the S1's used slots."""
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr)
    used = own.join(pool.select("pid", "src"), on="pid", how="left").group_by("q", "src").len().rename({"len": "used"})
    c = (pp.join(own.select("pid"), on="pid", how="anti").join(own.select("q", "pid"), on=["q", "pid"], how="anti")
           .join(s1, on="q", how="left").join(pool, on="pid", how="left")
           .join(used, on=["q", "src"], how="left").with_columns(pl.col("used").fill_null(0)))
    a, b = pl.col("a_core"), pl.col("b_core")
    ta, tb = a.str.split(" ").list.unique(), b.str.split(" ").list.unique()
    ini = b.str.len_chars().is_between(2, 3) & ~b.str.contains(" ")
    c = c.with_columns(a.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("_ini"))
    kind = (pl.when(a == b).then(pl.lit("exact"))
            .when(strip_spaced(a) == strip_spaced(b)).then(pl.lit("spelled_legal"))
            .when(ini & pl.col("_ini").str.starts_with(b)).then(pl.lit("initials"))
            .when(a.str.replace_all(" ", "") == b.str.replace_all(" ", "")).then(pl.lit("glued"))
            .when((ta.list.set_difference(tb).list.len() == 1) & (tb.list.set_difference(ta).list.len() == 1)
                  & tb.list.set_difference(ta).list.first().is_in(NOISE_FR)).then(pl.lit("noise_swap"))
            .otherwise(pl.lit("other")))
    return c.with_columns(kind.alias("kind"), pl.when(pl.col("src") == 2).then(5).otherwise(6).alias("cap")).with_columns((pl.col("used") < pl.col("cap")).alias("slot_free"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    xdir = sys.argv[2] if len(sys.argv) > 2 else None
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1, pool = texts(P, "test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    c = restore_candidates(pp, thr, s1, pool)
    c = c.join(feats(P, "test", c.select("q", "pid")), on=["q", "pid"], how="left").with_columns(
        ((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90)).alias("same_addr"))
    if xdir:
        xf = pl.read_parquet(P["work"] / xdir / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xfr"))
        c = c.join(xf, on=["q", "pid"], how="left")
    agg = [pl.len().alias("pairs"), pl.col("pid").n_unique().alias("records"), pl.col("p").mean().alias("mean_p"), (pl.col("p") >= 0.3).mean().alias("p>=0.3"),
           pl.col("slot_free").mean().alias("slot_free")] + ([pl.col("xfr").mean().alias("mean_xfr"), (pl.col("xfr") >= 0.5).mean().alias("xfr>=0.5")] if xdir else [])
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=220, fmt_str_lengths=36):
        for ctry in ("france", "us", "india"):
            print(f"\n== {ctry}: restore candidates of {name} (shortlisted, below the decision, pool record owned by nobody)")
            print(c.filter(pl.col("ctry") == ctry).group_by("kind", "same_addr").agg(*agg).sort("kind", "same_addr"))
        ex = c.filter((pl.col("ctry") == "france") & pl.col("same_addr") & (pl.col("kind") != "other"))
        print(f"\nFrance examples ({ex.height} strong-pattern pairs at the same address):")
        print(ex.sample(min(30, ex.height), seed=3).select("kind", "p", *(["xfr"] if xdir else []), "a_core", "b_core", "used", "cap"))
    # holdout: the same candidates where labels exist
    s1h, poolh = texts(P, "train")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    hthr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    ph = ph.join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi")
    lab = ph.select("q", "pid", "label")
    ch = restore_candidates(ph.drop("label"), hthr, s1h, poolh).join(lab, on=["q", "pid"], how="left")
    ch = ch.join(feats(P, "train", ch.select("q", "pid")), on=["q", "pid"], how="left").with_columns(
        ((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90)).alias("same_addr"))
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=220):
        print("\n== holdout (labelled): the same restore candidates, true share")
        print(ch.group_by("kind", "same_addr").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), pl.col("label").mean().alias("true_share"),
                                                   pl.col("label").filter(pl.col("p") >= 0.3).mean().alias("true_share_p>=0.3")).sort("kind", "same_addr"))


if __name__ == "__main__":
    main()
