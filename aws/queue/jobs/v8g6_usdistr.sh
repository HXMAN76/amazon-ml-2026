# v8 (CPU lane jobs2): does the US/India test output claim generator distractors? Claimed pool records carrying a location suffix
# (southside/eastgate/northside/midtown/greater/lakeside/westgate/riverside: 0% owned in train), by word and country, holdout against test, examples.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths()
LOC = ["southside", "eastgate", "northside", "midtown", "greater", "lakeside", "westgate", "riverside"]
def pool(split):
    d = pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "name1", "ctry"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("name1").alias("b_name"), "ctry") for s in (2, 3)])
    w = pl.col("b_name").str.split(" ")
    return d.with_columns(pl.coalesce([pl.when(w.list.contains(x)).then(pl.lit(x)) for x in LOC]).alias("loc"))
m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
tr = pool("train")
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
h = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
k = decision.assign_exclusive(h).filter(pl.col("p") >= thr).join(tr, on="pid", how="left")
print(f"holdout: kept pairs {k.height}; with a location-suffix record {k.filter(pl.col('loc').is_not_null()).height} (true {k.filter(pl.col('loc').is_not_null())['label'].sum()})")
print(f"train pool location-suffix records: {tr.filter(pl.col('loc').is_not_null()).height} of {tr.height} ({tr.filter(pl.col('loc').is_not_null()).height / tr.height:.4f})")
te = pool("test")
print(f"test pool location-suffix records: {te.filter(pl.col('loc').is_not_null()).height} of {te.height} ({te.filter(pl.col('loc').is_not_null()).height / te.height:.4f})")
pp = pl.read_parquet(P["work"] / "output" / "v8u_s28_AR" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
own = decision.assign_exclusive(pp).filter(pl.col("p") >= 0.72).join(te, on="pid", how="left")
s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "name1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"), pl.col("ctry").alias("s1_ctry"))
own = own.join(s1, on="q", how="left")
with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=50):
    print("test v8u_s28_AR: predicted pairs whose record carries a location suffix, by S1 country")
    print(own.group_by("s1_ctry").agg(pl.len().alias("pairs"), pl.col("loc").is_not_null().sum().alias("loc_pairs"), pl.col("p").filter(pl.col("loc").is_not_null()).mean().alias("mean_p")))
    x = own.filter(pl.col("loc").is_not_null())
    print(x.sample(min(20, x.height), seed=0).select("p", "s1_ctry", "a_name", "b_name"))
PY
