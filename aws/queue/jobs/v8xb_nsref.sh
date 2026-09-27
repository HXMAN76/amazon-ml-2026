# v8 fork B (lane jobs): refined namesake droplists relative to v8u_s28_ALL2 (else ALL): exact core (fb_ns_ref) or near-exact (fb_nsnear_ref) on another
# street, name shared by >= 11 France S1 with p < 0.9999, or by >= 6 with p < 0.99 (the cells with a French excess; >= 0.9999 has none).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, sys
import polars as pl
sys.path.insert(0, "src/scripts")
from ber import config, decision
from namesake_street import compare, rare_vocab, street
PB = 10_000_000
P = config.paths(); pq = P["parquet"] / "test"
thr = json.loads((P["work"] / "models" / "s28" / "config.json").read_text())["threshold"]
run = "v8u_s28_ALL2" if (P["work"] / "output" / "v8u_s28_ALL2" / "pair_p.parquet").exists() else "v8u_s28_ALL"
pa = pl.read_parquet(P["work"] / "output" / run / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
pp = pl.read_parquet(P["work"] / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).select("q", "pid", pl.col("p").alias("p_raw"))
ka = decision.assign_exclusive(pa).filter(pl.col("p") >= thr).select("q", "pid").join(pp, on=["q", "pid"], how="left")
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).filter(pl.col("ctry") == "france").select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"))
s1 = s1.with_columns(pl.len().over("a_core").alias("n_ns"))
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry"]).filter(pl.col("ctry") == "france").select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr")) for s in (2, 3)])
rare = rare_vocab(s1["a_addr"], pool["b_addr"])
d = ka.join(s1, on="q").join(pool, on="pid").filter(pl.col("n_ns") >= 6)
ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
d = d.with_columns((pl.col("a_core") == pl.col("b_core")).alias("exact"),
                   ((ta.list.set_difference(tb).list.len() + tb.list.set_difference(ta).list.len() <= 2) & (ta.list.set_intersection(tb).list.len() >= 1) & (pl.col("a_core") != pl.col("b_core"))).alias("near"))
d = d.filter(pl.col("exact") | pl.col("near"))
d = street(street(d, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
sel = (pl.col("st") == 1) & (((pl.col("n_ns") >= 11) & (pl.col("p_raw") < 0.9999)) | (pl.col("p_raw") < 0.99))
for nm, f in (("fb_ns_ref", pl.col("exact")), ("fb_nsnear_ref", pl.col("near"))):
    x = d.filter(sel & f)
    x.select("q", "pid").write_parquet(P["work"] / "v8x" / f"{nm}.parquet")
    print(f"{nm}: {x.height} pairs of {run}'s kept France pairs (n_ns>=11: {x.filter(pl.col('n_ns') >= 11).height})", flush=True)
    with pl.Config(tbl_rows=20, fmt_str_lengths=40, tbl_width_chars=200):
        print(x.sample(min(20, x.height), seed=0).select("p_raw", "n_ns", "a_core", "b_core", "a_addr", "b_addr"))
PY
