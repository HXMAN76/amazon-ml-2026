# v8 (CPU lane jobs2): coined aliases (no common name word) at the S1's exact address (same house number, address token-set >= 90) by probability band:
# pairs per 1,000 S1 in France, US, India (test, s28), and the holdout's true share of the same kind and band. France's rate against the US/India rate
# says whether the ones the 0.995 cut drops are true aliases (same rate) or extra decoys (excess).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths()
def feats(split, keys):
    fs = sorted(str(f) for f in (P["work"] / "features" / split).glob("part_*.parquet"))
    return pl.scan_parquet(fs).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset").join(keys.lazy(), on=["q", "pid"], how="semi").collect()
def names(split):
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b")) for s in (2, 3)])
    return s1, pool
def tag(d):
    ta, tb = pl.col("a").str.split(" ").list.unique(), pl.col("b").str.split(" ").list.unique()
    return d.with_columns((ta.list.set_intersection(tb).list.len() == 0).alias("coined"),
                          pl.col("p").cut([0.72, 0.9, 0.99, 0.995, 0.9999], left_closed=True).cast(pl.String).alias("band"))
s1, pool = names("test")
pp = pl.read_parquet(P["work"] / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
own = decision.assign_exclusive(pp).filter(pl.col("p") >= 0.72).join(s1, on="q", how="left").join(pool, on="pid", how="left")
own = tag(own).filter(pl.col("coined"))
own = own.join(feats("test", own.select("q", "pid")), on=["q", "pid"], how="left").filter((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
n1 = s1.group_by("ctry").len().rename({"len": "n_s1"})
t = own.group_by("ctry", "band").len().join(n1, on="ctry").with_columns((1000 * pl.col("len") / pl.col("n_s1")).alias("per_1k_S1")).sort("band", "ctry")
m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
s1h, poolh = names("train")
h = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
h = tag(decision.assign_exclusive(h).filter(pl.col("p") >= thr).join(s1h, on="q", how="left").join(poolh, on="pid", how="left")).filter(pl.col("coined"))
h = h.join(feats("train", h.select("q", "pid")), on=["q", "pid"], how="left").filter((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
hh = h.group_by("band").agg(pl.len().alias("holdout_pairs"), (1000 * pl.len() / hq.height).alias("holdout_per_1k_S1"), pl.col("label").mean().alias("holdout_true"))
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print("coined aliases at the S1's exact address, kept by s28 (exclusive, p >= 0.72), per 1,000 S1")
    print(t.join(hh, on="band", how="left").sort("band", "ctry"))
PY
