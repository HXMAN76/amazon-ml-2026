# v8 (CPU lane jobs2): exact-name candidates whose pool record has an empty address (the largest block of s28's lost true pairs): does the pool
# show other businesses with that name (same core name, non-empty address, another house number)? True share and s28's p by that evidence.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths(); m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
h = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
own = decision.assign_exclusive(h).select("q", "pid", pl.lit(True).alias("owner"))
h = h.join(own, on=["q", "pid"], how="left").with_columns(pl.col("owner").fill_null(False))
pq = P["parquet"] / "train"
hn = lambda c: pl.col(c).str.extract(r"(\d+)", 1)
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), hn("addr").alias("ha"))
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr")) for s in (2, 3)])
cnt_s1 = s1.group_by("a_core").len().rename({"a_core": "b_core", "len": "n_s1_name"})
d = h.join(s1, on="q", how="left").join(pool, on="pid", how="left").filter((pl.col("a_core") == pl.col("b_core")) & (pl.col("b_addr") == ""))
named = pool.filter(pl.col("b_addr") != "").join(d.select("b_core").unique(), on="b_core", how="semi").select("b_core", hn("b_addr").alias("hp"))
ev = d.select("q", "pid", "b_core", "ha").join(named, on="b_core", how="left").group_by("q", "pid").agg(
    (pl.col("hp").is_not_null() & (pl.col("hp") == pl.col("ha"))).sum().alias("n_here"), (pl.col("hp").is_not_null() & (pl.col("hp") != pl.col("ha"))).sum().alias("n_other"))
d = d.join(ev, on=["q", "pid"], how="left").join(cnt_s1, on="b_core", how="left")
d = d.with_columns(pl.when(pl.col("n_other") == 0).then(pl.lit("0")).when(pl.col("n_other") <= 2).then(pl.lit("1-2")).otherwise(pl.lit("3+")).alias("other_named"),
                   pl.when(pl.col("n_s1_name") == 1).then(pl.lit("1")).when(pl.col("n_s1_name") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")).alias("s1_named"),
                   pl.when(pl.col("n_here") == 0).then(pl.lit("0")).otherwise(pl.lit("1+")).alias("here_named"))
with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=220):
    print(f"holdout exact-name candidates with an empty pool address: {d.height}, true {d['label'].sum()}, kept {d.filter(pl.col('owner') & (pl.col('p') >= thr)).height}")
    print(d.group_by("s1_named", "here_named", "other_named").agg(pl.len().alias("n"), pl.col("label").mean().alias("true_share"), pl.col("p").mean().alias("mean_p"),
          ((pl.col("p") >= thr) & pl.col("owner")).mean().alias("kept_share"), ((pl.col("p") < thr) & (pl.col("label") == 1)).sum().alias("true_not_kept"),
          ((pl.col("p") >= thr) & pl.col("owner") & (pl.col("label") == 0)).sum().alias("wrong_kept")).sort("s1_named", "here_named", "other_named"))
    x = d.filter(pl.col("owner"))
    print("owner pairs only (after exclusive assignment), by p band and other_named:")
    print(x.with_columns(pl.col("p").cut([0.1, 0.3, 0.5, 0.72, 0.9]).cast(pl.String).alias("pz")).group_by("pz", "other_named").agg(pl.len().alias("n"), pl.col("label").mean().alias("true_share")).sort("pz", "other_named"))
PY
