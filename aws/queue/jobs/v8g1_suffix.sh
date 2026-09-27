# v8 (CPU lane jobs2): pool records carrying a generator-injected suffix word (pool-to-S1 frequency ratio in the thousands: US southside/eastgate/...,
# France participations/holding/distribution/international). Train: are they true copies (labels)? holdout: what s28 does with them.
# Test: how many France/US suffix records s28 / v8u_s28_AR claim, and the p of the unclaimed ones.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths()
SUF = {"us": ["southside", "eastgate", "northside", "midtown", "greater", "lakeside", "westgate", "riverside"],
       "france": ["participations", "holding", "distribution", "international"], "india": ["southside", "eastgate", "northside", "midtown", "greater", "lakeside", "westgate", "riverside"]}
def pool(split):
    return pl.concat([pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["rid", "core1", "name1", "ctry"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name"), "ctry") for s in (2, 3)])
def suffix(d):
    w = pl.col("b_name").str.split(" ")
    return d.with_columns(pl.coalesce([pl.when(w.list.contains(x)).then(pl.lit(x)) for x in sorted({x for v in SUF.values() for x in v})]).alias("suf"))
tr = suffix(pool("train"))
lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").select((pl.col("src").cast(pl.Int64) * PB + pl.col("other_rid").cast(pl.Int64)).alias("pid")).unique().with_columns(pl.lit(True).alias("owned"))
tr = tr.join(lab, on="pid", how="left").with_columns(pl.col("owned").fill_null(False))
with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=50):
    print("train pool: share owned by some S1 (a true copy), by suffix word and country")
    print(tr.group_by("ctry", "suf").agg(pl.len().alias("records"), pl.col("owned").mean().alias("owned_share")).sort("ctry", "suf"))
    print(tr.filter(pl.col("suf").is_not_null()).sample(12, seed=0).select("ctry", "owned", "b_name"))
    m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    h = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    ex = decision.assign_exclusive(h).filter(pl.col("p") >= thr).select("q", "pid", pl.lit(True).alias("kept"))
    h = h.join(ex, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False)).join(tr.select("pid", "suf"), on="pid", how="left")
    print("holdout candidates with a suffix record: true pairs kept / not kept, wrong pairs kept")
    print(h.with_columns(pl.col("suf").is_not_null().alias("has_suf")).group_by("has_suf").agg(((pl.col("label") == 1) & pl.col("kept")).sum().alias("true_kept"),
          ((pl.col("label") == 1) & ~pl.col("kept")).sum().alias("true_not_kept"), ((pl.col("label") == 0) & pl.col("kept")).sum().alias("wrong_kept"), pl.len().alias("cands")))
    te = suffix(pool("test"))
    for run in ("s28", "v8u_s28_AR"):
        pp = pl.read_parquet(P["work"] / "output" / run / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
        own = decision.assign_exclusive(pp)
        best = own.select("pid", "p")
        t = te.join(best, on="pid", how="left").with_columns((pl.col("p") >= 0.72).fill_null(False).alias("claimed"))
        print(f"test {run}: suffix records claimed (p >= 0.72 after exclusive assignment), by country and suffix; p of the best S1 when unclaimed")
        print(t.group_by("ctry", pl.col("suf").is_not_null().alias("has_suf")).agg(pl.len().alias("records"), pl.col("claimed").mean().alias("claimed_share"),
              pl.col("p").is_null().mean().alias("no_candidate"), pl.col("p").filter(~pl.col("claimed")).mean().alias("mean_p_unclaimed")).sort("ctry", "has_suf"))
    t = t.join(own.select("pid", "q"), on="pid", how="left")
    s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "name1"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"))
    x = t.filter((pl.col("ctry") == "france") & pl.col("suf").is_not_null()).join(s1, on="q", how="left")
    print("France suffix records: examples (claimed, then unclaimed with a candidate)")
    print(x.filter(pl.col("claimed")).sample(12, seed=1).select("p", "a_name", "b_name"))
    print(x.filter(~pl.col("claimed") & pl.col("p").is_not_null()).sample(12, seed=1).select("p", "a_name", "b_name"))
PY
