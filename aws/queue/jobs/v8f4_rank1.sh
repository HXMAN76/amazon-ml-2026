# v8 (CPU lane jobs2): rank-dependent threshold on s28: an S1's best owned pair kept at p >= t1, its other pairs at p >= t2 (an empty list scores 0
# when the S1 has a match, so the first pick may deserve a lower bar). Tuned out-of-fold, paired bootstrap on the locked holdout.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
P = config.paths(); m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
def prep(df):
    ex = decision.assign_exclusive(df.with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)))
    return ex.with_columns(pl.col("p").rank("ordinal", descending=True).over("q").alias("r"))
oof = prep(pl.read_parquet(m / "oof_tune.parquet")); nto = oof.select("q").unique().join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
hold = prep(pl.read_parquet(m / "holdout_pred.parquet").join(hq, on="q", how="semi")); nth = hq.join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
keep = lambda d, t1, t2: d.filter(((pl.col("r") == 1) & (pl.col("p") >= t1)) | ((pl.col("r") > 1) & (pl.col("p") >= t2)))
res = []
for t1 in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72):
    for t2 in (0.68, 0.72, 0.76, 0.80):
        res.append((t1, t2, decision.macro_f05(keep(oof, t1, t2), nto)))
res.sort(key=lambda x: -x[2])
print("out-of-fold best 8:", [(a, b, round(c, 6)) for a, b, c in res[:8]], "; base", round([c for a, b, c in res if a == 0.72 and b == 0.72][0], 6))
base = decision.per_entity_f05(keep(hold, thr, thr), nth).sort("q")["f"].to_numpy()
for t1, t2, _ in res[:3]:
    new = decision.per_entity_f05(keep(hold, t1, t2), nth).sort("q")["f"].to_numpy()
    d, lo, hi = decision.paired_bootstrap_delta(base, new)
    print(f"holdout t1 {t1} t2 {t2}: {new.mean():.6f} against {base.mean():.6f}, delta {d:+.6f} [{lo:+.6f}, {hi:+.6f}]")
PY
