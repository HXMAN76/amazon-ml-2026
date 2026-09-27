# v8 (CPU lane jobs2): where s28 still loses on the labelled holdout (US/India): blocking misses (true pairs never proposed) against matcher
# errors (wrong pairs kept, true candidates not kept), as the macro F0.5 gained by fixing each one alone; by country.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<PY
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
P = config.paths(); m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
ctry = pl.read_parquet(P["parquet"] / "train" / "source1.parquet", columns=["rid", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "ctry")
hold = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
pred = decision.assign_exclusive(hold).filter(pl.col("p") >= thr)
tc = hold.filter(pl.col("label") == 1).select("q", "pid", "label")
for c in ("all", "us", "india"):
    qs = hq if c == "all" else hq.join(ctry.filter(pl.col("ctry") == c), on="q", how="semi")
    nt = qs.join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    pr, t = pred.join(qs, on="q", how="semi"), tc.join(qs, on="q", how="semi")
    ncand = qs.join(t.group_by("q").len().rename({"len": "n_true"}), on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    f = decision.macro_f05(pr, nt)
    f_fp = decision.macro_f05(pr.filter(pl.col("label") == 1), nt)
    f_fn = decision.macro_f05(pl.concat([pr.select("q", "pid", "label"), t.join(pr, on=["q", "pid"], how="anti")]), nt)
    f_blk = decision.macro_f05(pr, ncand)
    f_orc = decision.macro_f05(t, nt)
    print(f"{c}: S1 {qs.height}; true pairs {int(nt['n_true'].sum())}, of them candidates {t.height} (pair recall {t.height / max(1, nt['n_true'].sum()):.4f}); "
          f"kept {pr.height}, wrong {pr.filter(pl.col('label') == 0).height}, true candidates not kept {t.join(pr, on=['q', 'pid'], how='anti').height}")
    print(f"  F0.5 {f:.6f}; oracle on candidates {f_orc:.6f}; gain if no wrong pair kept {f_fp - f:+.6f}; if every true candidate kept {f_fn - f:+.6f}; "
          f"if blocking missed nothing {f_blk - f:+.6f}")
PY
