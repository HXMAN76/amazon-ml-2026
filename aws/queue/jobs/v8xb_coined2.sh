# v8 fork B (lane jobs): plain coined pool names (one 6+ letter word outside the country's S1 vocabulary, no alias marker) at the S1's street and
# house number, s28 raw kept pairs: per 1,000 S1 France vs US vs India by band (<0.995, 0.995-0.9999, >=0.9999), holdout true share; France pairs
# of that cell dropped by v8u_s28_ALL2 (else ALL) per band; addlist fb_coined_hi = those at p >= 0.995.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, sys
import numpy as np, polars as pl
sys.path.insert(0, "src/scripts")
from ber import config, decision
from ber.split import holdout_q
from namesake_street import compare, house, rare_vocab, street
PB = 10_000_000
P = config.paths()
BAND = pl.when(pl.col("p") < 0.995).then(pl.lit("a<0.995")).when(pl.col("p") < 0.9999).then(pl.lit("b0.995-0.9999")).otherwise(pl.lit("c>=0.9999"))
def load(split):
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry", "has_alias"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"), pl.col("ctry").alias("ctry_b"), "has_alias") for s in (2, 3)])
    return s1, pool
def cell(pred, s1, pool, c):
    s1c, poolc = s1.filter(pl.col("ctry") == c), pool.filter(pl.col("ctry_b") == c)
    vocab = s1c["a_core"].str.split(" ").explode().unique().drop_nulls().implode()
    d = pred.join(s1c, on="q").join(poolc, on="pid").filter(pl.col("b_core").str.contains(r"^[a-z]{6,}$") & ~pl.col("b_core").is_in(vocab) & ~pl.col("has_alias"))
    rare = rare_vocab(s1c["a_addr"], poolc["b_addr"])
    d = street(street(d, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
    d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())), house(pl.col("a_addr")).alias("a_h"), house(pl.col("b_addr")).alias("b_h"))
    return d.filter((pl.col("st") == 0) & pl.col("a_h").is_not_null() & (pl.col("a_h") == pl.col("b_h"))).with_columns(BAND.alias("band"), pl.lit(c).alias("ctry"))
thr = json.loads((P["work"] / "models" / "s28" / "config.json").read_text())["threshold"]
s1, pool = load("test")
pp = pl.read_parquet(P["work"] / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
raw = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
T = pl.concat([cell(raw, s1, pool, c).select("q", "pid", "p", "band", "ctry", "a_core", "b_core") for c in ("france", "us", "india")])
n1 = {c: s1.filter(pl.col("ctry") == c).height for c in ("france", "us", "india")}
w = T.group_by("band", "ctry").len().pivot(on="ctry", index="band", values="len").fill_null(0)
for c in ("france", "us", "india"):
    w = w.with_columns((1000 * pl.col(c) / n1[c]).alias(f"{c[:2]}_per1k"))
w = w.with_columns(((pl.col("fr_per1k") - pl.max_horizontal("us_per1k", "in_per1k")) / pl.col("fr_per1k")).alias("decoy_vs_max"),
                   ((pl.col("fr_per1k") - pl.min_horizontal("us_per1k", "in_per1k")) / pl.col("fr_per1k")).alias("decoy_vs_min"))
hthr = json.loads((P["work"] / "models" / "s28" / "holdout.json").read_text())["stack_threshold"]
s1h, poolh = load("train")
hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
ph = pl.read_parquet(P["work"] / "models" / "s28" / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
kh = decision.assign_exclusive(ph).filter(pl.col("p") >= hthr).select("q", "pid", "p", "label")
H = pl.concat([cell(kh, s1h, poolh, c) for c in ("us", "india")]).group_by("band").agg(pl.len().alias("hold_pairs"), pl.col("label").mean().alias("hold_true"))
run = "v8u_s28_ALL2" if (P["work"] / "output" / "v8u_s28_ALL2" / "pair_p.parquet").exists() else "v8u_s28_ALL"
pa = pl.read_parquet(P["work"] / "output" / run / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
ka = decision.assign_exclusive(pa).filter(pl.col("p") >= thr).select("q", "pid")
fr = T.filter(pl.col("ctry") == "france").join(ka, on=["q", "pid"], how="anti")
with pl.Config(tbl_rows=20, tbl_width_chars=220, fmt_str_lengths=40):
    print("plain coined names at the S1's street + house number, s28 raw kept, per 1,000 S1:")
    print(w.join(H, on="band", how="left").join(fr.group_by("band").len().rename({"len": f"fr_dropped_by_{run}"}), on="band", how="left").sort("band"))
    hi = fr.filter(pl.col("p") >= 0.995)
    hi.select("q", "pid").write_parquet(P["work"] / "v8x" / "fb_coined_hi.parquet")
    print(f"fb_coined_hi: {hi.height} pairs (p >= 0.995, dropped by {run}); examples")
    print(hi.sample(min(20, hi.height), seed=0).select("p", "a_core", "b_core"))
    print(fr.filter(pl.col("p") < 0.995).sample(min(10, fr.filter(pl.col('p') < 0.995).height), seed=0).select("p", "a_core", "b_core"))
PY
