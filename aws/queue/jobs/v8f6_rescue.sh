# v8 (CPU lane jobs2): recall rescue for records the shortlist never showed the stack: exact core name (or core + one extra word), pool address
# empty, pair not among s28's holdout candidates, record unclaimed. True share by name uniqueness, and the holdout macro F0.5 change of adding them.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths(); m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
pq = P["parquet"] / "train"
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
lab = pl.read_parquet(pq / "labels.parquet").select(pl.col("s1_rid").cast(pl.Int64).alias("q"), (pl.col("src").cast(pl.Int64) * PB + pl.col("other_rid").cast(pl.Int64)).alias("pid"), pl.lit(1).alias("label"))
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("core"), "ctry")
s1 = s1.join(s1.group_by("core", "ctry").len().rename({"len": "n_s1_name"}), on=["core", "ctry"], how="left")
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), "addr", "ctry") for s in (2, 3)])
empty = pool.filter(pl.col("addr") == "")
print(f"train pool records {pool.height}, empty address {empty.height} ({empty.height / pool.height:.4f})")
cand = pl.concat([pl.read_parquet(m / "oof_tune.parquet").select("q", "pid", "p"), pl.read_parquet(m / "holdout_pred.parquet").select("q", "pid", "p")]).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
claimed = decision.assign_exclusive(cand).filter(pl.col("p") >= thr).select("pid")
hold = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
kept = decision.assign_exclusive(hold).filter(pl.col("p") >= thr).join(lab, on=["q", "pid"], how="left").with_columns(pl.col("label").fill_null(0))
hs = s1.join(hq, on="q", how="semi")
# exact core, and core + one extra word (either end)
ex = hs.join(empty, left_on=["core", "ctry"], right_on=["b_core", "ctry"], how="inner").with_columns(pl.lit("exact").alias("rel"), pl.lit("").alias("w"))
e2 = empty.with_columns(pl.col("b_core").str.split(" ").alias("_t")).filter(pl.col("_t").list.len() >= 2)
e2 = pl.concat([e2.with_columns(pl.col("_t").list.slice(0, pl.col("_t").list.len() - 1).list.join(" ").alias("core"), pl.col("_t").list.last().alias("w")),
                e2.with_columns(pl.col("_t").list.slice(1).list.join(" ").alias("core"), pl.col("_t").list.first().alias("w"))]).drop("_t")
plus = hs.join(e2, on=["core", "ctry"], how="inner").with_columns(pl.lit("plus1").alias("rel"))
r = pl.concat([ex.select("q", "pid", "ctry", "n_s1_name", "rel", "w"), plus.select("q", "pid", "ctry", "n_s1_name", "rel", "w")])
r = r.join(hold.select("q", "pid"), on=["q", "pid"], how="anti").join(claimed, on="pid", how="anti").join(lab, on=["q", "pid"], how="left").with_columns(pl.col("label").fill_null(0))
r = r.with_columns(pl.col("q").n_unique().over("pid").alias("n_q_pid"))
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(r.group_by("rel", pl.col("n_s1_name").clip(1, 4).alias("s1_named")).agg(pl.len().alias("pairs"), pl.col("label").mean().alias("true_share"), pl.col("label").sum().alias("true")).sort("rel", "s1_named"))
    print("plus1 by extra word (top 25 by pairs, unique S1 name):")
    print(r.filter((pl.col("rel") == "plus1") & (pl.col("n_s1_name") == 1)).group_by("w").agg(pl.len().alias("pairs"), pl.col("label").mean().alias("true_share")).sort("pairs", descending=True).head(25))
hth = hq.join(pl.read_parquet(pq / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64)), on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
base = decision.per_entity_f05(kept, hth).sort("q")["f"].to_numpy()
used = kept.with_columns((pl.col("pid") // PB).alias("src")).group_by("q", "src").len().rename({"len": "used"})
for name, f in (("exact, unique name", (pl.col("rel") == "exact") & (pl.col("n_s1_name") == 1)),
                ("exact, name on <= 2 S1", (pl.col("rel") == "exact") & (pl.col("n_s1_name") <= 2)),
                ("exact or plus1, unique name", pl.col("n_s1_name") == 1)):
    add = r.filter(f & (pl.col("n_q_pid") == 1)).unique(["q", "pid"]).with_columns((pl.col("pid") // PB).alias("src")).join(used, on=["q", "src"], how="left").with_columns(pl.col("used").fill_null(0))
    add = add.with_columns(pl.col("pid").rank("ordinal").over(["q", "src"]).alias("_rk")).filter(pl.col("_rk") + pl.col("used") <= pl.when(pl.col("src") == 2).then(5).otherwise(6))
    new = decision.per_entity_f05(pl.concat([kept.select("q", "pid", "label"), add.select("q", "pid", "label")], how="vertical_relaxed"), hth).sort("q")["f"].to_numpy()
    d, lo, hi = decision.paired_bootstrap_delta(base, new)
    print(f"add {name}: {add.height} pairs, true share {add['label'].mean():.3f}; holdout {new.mean():.6f} against {base.mean():.6f}, delta {d:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)
PY
