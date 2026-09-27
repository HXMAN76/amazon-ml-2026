"""Where the France-aware cross-encoder disagrees with the old one, who is right? Usage: python src/scripts/xfr_slots.py MODEL XDIR [OLD_XDIR]
For France's predicted pairs of MODEL (and the US as the reference for the fit's bias): kind of name relation x new score below 0.5 x old score below
0.5, with pairs and the slot-limit decoy share (france_variants.decoy_share). A cell that the new model rejects and the old one accepted should
show a clearly higher decoy share than its accepted twin cell, and than the same cell in the US, if the new model is finding decoys."""

import json
import sys

import polars as pl

from ber import config, decision
from france_variants import decoy_share
from word_swap import flag, tok_df

PID_BASE = 10_000_000
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
    xn = pl.read_parquet(P["work"] / xdir / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("x_new"))
    xo = pl.read_parquet(P["work"] / old / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("x_old"))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"),
                                 pl.col("b_core").str.split(" ").list.set_difference(pl.col("a_core").str.split(" ")).list.first().alias("xb"))
    d = d.join(xo, on=["q", "pid"], how="left").join(xn, on=["q", "pid"], how="left")
    kind = (pl.when(pl.col("core_eq")).then(pl.lit("exact")).when(pl.col("swap") & pl.col("xb").is_in(list(NOISE_FR))).then(pl.lit("swap_noise"))
            .when(pl.col("swap")).then(pl.lit("swap")).otherwise(pl.lit("other")))
    d = d.with_columns(kind.alias("kind"), (pl.col("x_new") < 0.5).alias("new_rej"), (pl.col("x_old") < 0.5).alias("old_rej"))
    for c in ("france", "us"):
        dc = d.filter(pl.col("ctry") == c)
        ex = dc.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
        slots = s1.filter(pl.col("ctry") == c).select("q").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        rows = []
        for (kd, nr, orj), g in dc.filter(pl.col("x_new").is_not_null() & pl.col("x_old").is_not_null()).group_by("kind", "new_rej", "old_rej"):
            rows.append([kd, nr, orj, g.height, round(float(g["p"].mean()), 4), *decoy_share(g, slots)])
        with pl.Config(tbl_rows=30, tbl_width_chars=200):
            print(f"\n== {c} (scored pairs only; the US needs {old} and {xdir} scores for US pairs, else it is empty)")
            print(pl.DataFrame(rows, schema=["kind", "new_below_0.5", "old_below_0.5", "pairs", "mean_p", "decoy_S2", "decoy_S3"], orient="row").sort("kind", "new_below_0.5", "old_below_0.5"))


if __name__ == "__main__":
    main()
