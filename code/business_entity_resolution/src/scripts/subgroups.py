"""Is the stacked model miscalibrated in some sub-group where labels exist? Usage: python src/scripts/subgroups.py MODEL
Discovery on MODEL's out-of-fold tuning pairs (oof_tune.parquet), confirmation on the locked holdout (holdout_pred.parquet); features from the
model's train chunks (stack<tag>/train). Two searches:
  drop:    predicted pairs (exclusive assignment, threshold) by sub-group; a group whose true share is below about 0.70 is worth dropping.
  restore: pairs below the threshold whose pool record nobody owns, by sub-group; a group whose true share is above about 0.75 is worth adding.
Sub-groups: probability zone x cross-encoder disagreement (xs, xs4 below 0.5) x empty pool address x house-number agreement x name similarity.
For each candidate group found out-of-fold, prints the holdout macro F0.5 change of applying it (paired bootstrap)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

GROUPS = ["pz", "xs_lo", "xs4_lo", "b_empty", "house", "name"]


def feats(P, tag: str) -> pl.LazyFrame:
    files = sorted(str(f) for f in (P["work"] / f"stack{tag}" / "train").glob("chunk_*.parquet"))
    have = set(pl.scan_parquet(files[0]).collect_schema().names())
    cols = [c for c in ("xs", "xs4", "addr_b_empty", "house_eq", "name_tset") if c in have]
    print("feature columns used:", cols, flush=True)
    return pl.scan_parquet(files).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), *cols)


def groups(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(
        pl.col("p").cut([0.1, 0.3, 0.5, 0.72, 0.8, 0.9, 0.99], left_closed=True).cast(pl.String).alias("pz"),
        (pl.col("xs") < 0.5).fill_null(False).alias("xs_lo") if "xs" in d.columns else pl.lit(False).alias("xs_lo"),
        (pl.col("xs4") < 0.5).fill_null(False).alias("xs4_lo") if "xs4" in d.columns else pl.lit(False).alias("xs4_lo"),
        (pl.col("addr_b_empty") > 0.5).fill_null(False).alias("b_empty") if "addr_b_empty" in d.columns else pl.lit(False).alias("b_empty"),
        (pl.col("house_eq") > 0.5).fill_null(False).alias("house") if "house_eq" in d.columns else pl.lit(False).alias("house"),
        pl.col("name_tset").cut([60, 80, 95], left_closed=True).cast(pl.String).fill_null("na").alias("name") if "name_tset" in d.columns else pl.lit("na").alias("name"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s28"
    P = config.paths()
    mdl = P["work"] / "models" / name
    cfg = json.loads((mdl / "config.json").read_text())
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    F = feats(P, cfg.get("tag", ""))
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))

    def prep(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        ex = decision.assign_exclusive(df)
        owned = ex.filter(pl.col("p") >= thr)
        cand = df.join(owned.select("pid"), on="pid", how="anti").join(owned.select("q", "pid"), on=["q", "pid"], how="anti")
        keys = pl.concat([owned.select("q", "pid"), cand.select("q", "pid")])
        f = F.join(keys.lazy(), on=["q", "pid"], how="semi").collect()
        return groups(owned.join(f, on=["q", "pid"], how="left")), groups(cand.join(f, on=["q", "pid"], how="left"))

    oof = pl.read_parquet(mdl / "oof_tune.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    po, co = prep(oof)
    drop = po.group_by(GROUPS).agg(pl.len().alias("n"), pl.col("label").mean().alias("true")).filter((pl.col("n") >= 200) & (pl.col("true") < 0.70)).sort("true")
    rest = co.filter(pl.col("p") >= 0.1).group_by(GROUPS).agg(pl.len().alias("n"), pl.col("label").mean().alias("true")).filter((pl.col("n") >= 200) & (pl.col("true") > 0.75)).sort("true", descending=True)
    with pl.Config(tbl_rows=30, tbl_cols=10, tbl_width_chars=200):
        print(f"out-of-fold: {po.height} predicted pairs, {co.height} candidates below the decision")
        print("drop candidates (true share < 0.70):"); print(drop)
        print("restore candidates (true share > 0.75):"); print(rest)
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    hold = pl.read_parquet(mdl / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    nt = hq.join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    ph, ch = prep(hold)
    base = decision.per_entity_f05(ph, nt).sort("q")["f"].to_numpy()
    for kind, tab, frame in (("drop", drop, ph), ("restore", rest, ch)):
        for g in tab.iter_rows(named=True):
            m = pl.all_horizontal([pl.col(c) == g[c] for c in GROUPS])
            sel = frame.filter(m).select("q", "pid", "p", "label")
            pred = ph.join(sel.select("q", "pid"), on=["q", "pid"], how="anti") if kind == "drop" else pl.concat([ph.select(sel.columns), sel.unique("pid")], how="vertical_relaxed")
            new = decision.per_entity_f05(pred, nt).sort("q")["f"].to_numpy()
            d, lo, hi = decision.paired_bootstrap_delta(base, new)
            print(f"{kind} {[g[c] for c in GROUPS]} (oof n {g['n']}, true {g['true']:.3f}): holdout pairs {sel.height}, true {sel['label'].mean() if sel.height else float('nan'):.3f}, "
                  f"delta {d:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)


if __name__ == "__main__":
    main()
