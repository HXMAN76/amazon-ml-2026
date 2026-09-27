"""Where is recall lost, stage by stage, and how much F0.5 does each loss cost? Locked holdout, one stacked model.

Usage: python src/scripts/attrition.py MODEL [BASE]      (defaults: s17 on the first stage v7)
Prints (1) the fate of every true pair: never proposed / proposed but cut by the shortlist / shortlisted but below the threshold /
lost to another S1 in the exclusive assignment / found; (2) the same fates by pool-record type (empty address, source, country) and by
number of true matches of the S1; (3) the per-S1 F0.5 loss split into missing matches, wrong matches and singleton false alarms;
(4) the calibration of the final probability near the threshold and a threshold sweep (precision, recall, macro F0.5).
"""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from ber.stages.stack import load_p1, shortlist

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    base = sys.argv[2] if len(sys.argv) > 2 else "v7"
    P = config.paths()
    mdl = P["work"] / "models" / name
    thr = json.loads((mdl / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").with_columns(
        (pl.col("src").cast(pl.Int64) * PID_BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").cast(pl.Int64).alias("q")).select("q", "pid")
    lab = lab.join(hq, on="q", how="semi")
    n_true = lab.group_by("q").len().rename({"len": "n_true"})
    nt = hq.join(n_true, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))

    cand = load_p1("train", base).join(hq, on="q", how="semi").select("q", "pid", "p")
    short = shortlist(cand.with_columns(pl.col("p")), config.load()["stack"]).select("q", "pid")
    pred = pl.read_parquet(mdl / "holdout_pred.parquet").select("q", "pid", "p", "label").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pred)
    found = own.filter(pl.col("p") >= thr).select("q", "pid").with_columns(pl.lit(1).alias("found"))

    t = (lab.join(cand.select("q", "pid").with_columns(pl.lit(1).alias("in_cand")), on=["q", "pid"], how="left")
            .join(short.with_columns(pl.lit(1).alias("in_short")), on=["q", "pid"], how="left")
            .join(pred.rename({"p": "p_final"}), on=["q", "pid"], how="left")
            .join(own.select("q", "pid").with_columns(pl.lit(1).alias("owned")), on=["q", "pid"], how="left")
            .join(found, on=["q", "pid"], how="left"))
    t = t.with_columns(
        pl.when(pl.col("in_cand").is_null()).then(pl.lit("1 never proposed"))
          .when(pl.col("in_short").is_null()).then(pl.lit("2 cut by shortlist"))
          .when(pl.col("found").is_not_null()).then(pl.lit("6 found"))
          .when(pl.col("p_final") < thr).then(pl.lit("4 below threshold"))
          .otherwise(pl.lit("5 lost to another S1 (exclusive)")).alias("fate"))
    n = t.height
    print(f"model {name}, threshold {thr:.2f}, true pairs {n}, holdout S1 {hq.height}")
    print(t.group_by("fate").agg(pl.len().alias("n"), (pl.len() / n).alias("share")).sort("fate"))
    bt = t.filter(pl.col("fate") == "4 below threshold")
    print("below threshold, final p:", bt.select(pl.col("p_final").quantile(q).alias(f"q{q}") for q in (0.1, 0.5, 0.9)).to_dicts()[0],
          " share within 0.3 of threshold:", float((bt["p_final"] > thr - 0.3).mean()))

    cols = ["rid", "addr", "ctry", "core1"]
    pool = pl.concat([pl.read_parquet(P["parquet"] / "train" / f"source{s}.parquet", columns=cols)
                        .with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid")
    s1 = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "addr"]).rename({"rid": "q", "addr": "s1_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    t = t.join(pool, on="pid", how="left").join(s1, on="q", how="left").join(n_true, on="q", how="left").with_columns(
        (pl.col("addr").str.len_chars() == 0).alias("pool_addr_empty"), (pl.col("s1_addr").str.len_chars() == 0).alias("s1_addr_empty"),
        pl.when(pl.col("n_true") == 1).then(pl.lit("n_true 1")).when(pl.col("n_true") == 2).then(pl.lit("n_true 2"))
          .when(pl.col("n_true") <= 4).then(pl.lit("n_true 3-4")).otherwise(pl.lit("n_true 5+")).alias("ntb"))
    for key in ("pool_addr_empty", "s1_addr_empty", "src", "ctry", "ntb"):
        g = t.group_by(key).agg(pl.len().alias("pairs"), (pl.col("fate") == "6 found").mean().alias("found"),
                                (pl.col("fate") == "1 never proposed").mean().alias("never"), (pl.col("fate") == "2 cut by shortlist").mean().alias("cut"),
                                (pl.col("fate") == "4 below threshold").mean().alias("below"),
                                (pl.col("fate") == "5 lost to another S1 (exclusive)").mean().alias("lost_excl")).sort("pairs", descending=True)
        print(f"\nby {key}\n{g}")
    miss = t.filter(pl.col("fate") != "6 found")
    print("\nshare of all missed pairs with empty pool address:", float(miss["pool_addr_empty"].mean()), " with empty S1 address:", float(miss["s1_addr_empty"].mean()))

    per = decision.per_entity_f05(own.filter(pl.col("p") >= thr), nt)
    tp = found.join(lab.with_columns(pl.lit(1).alias("is_true")), on=["q", "pid"], how="inner").group_by("q").len().rename({"len": "tp"})
    npred = found.group_by("q").len().rename({"len": "npred"})
    e = per.join(nt, on="q", how="left").join(tp, on="q", how="left").join(npred, on="q", how="left").with_columns(
        pl.col("tp").fill_null(0), pl.col("npred").fill_null(0))
    nt_col = "n_true"
    e = e.with_columns(
        pl.when((pl.col(nt_col) == 0) & (pl.col("npred") == 0)).then(pl.lit("a empty and correct"))
          .when(pl.col(nt_col) == 0).then(pl.lit("b singleton false alarm (no true match, predicted some)"))
          .when(pl.col("npred") == 0).then(pl.lit("c has matches, predicted none"))
          .when((pl.col("tp") == pl.col(nt_col)) & (pl.col("npred") == pl.col("tp"))).then(pl.lit("d perfect"))
          .when(pl.col("npred") == pl.col("tp")).then(pl.lit("e correct but incomplete (missing only)"))
          .when(pl.col("tp") == pl.col(nt_col)).then(pl.lit("f complete but with wrong extras"))
          .otherwise(pl.lit("g both missing and wrong")).alias("kind"))
    tot = hq.height
    g = e.group_by("kind").agg(pl.len().alias("s1"), (pl.len() / tot).alias("share_of_s1"), ((1 - pl.col("f")).sum() / tot).alias("loss_F05")).sort("kind")
    print(f"\nper-S1 F0.5 loss (macro F0.5 {float(e['f'].mean()):.5f}, total loss {float((1 - e['f']).sum() / tot):.5f})\n{g}")

    q = pred.filter(pl.col("p") >= thr - 0.5).join(lab.with_columns(pl.lit(1).alias("y")), on=["q", "pid"], how="left").with_columns(pl.col("y").fill_null(0))
    bins = np.array([0.0, 0.1, 0.3, 0.5, thr - 0.05, thr + 0.05, 0.9, 0.97, 1.01])
    cal = q.with_columns(pl.col("p").cut(list(bins[1:-1])).alias("bin")).group_by("bin").agg(pl.len().alias("pairs"), pl.col("y").mean().alias("true_share")).sort("bin")
    print(f"\ncalibration of the final p (threshold {thr:.2f})\n{cal}")
    print("\nthreshold sweep (exclusive assignment)")
    ntrue_all = float(lab.height)
    for th in (0.3, 0.4, 0.5, 0.6, 0.65, thr, 0.8, 0.9):
        f = own.filter(pl.col("p") >= th)
        tps = f.join(lab, on=["q", "pid"], how="semi").height
        print(f"  thr {th:.2f}: precision {tps / max(f.height, 1):.4f} recall {tps / ntrue_all:.4f} macro F0.5 {decision.macro_f05(f, nt):.5f}")


if __name__ == "__main__":
    main()
