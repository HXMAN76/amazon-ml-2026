# v9 (CPU lane jobs2): FIN-kept "words added" pairs at the S1's exact address: which word is added, by raw s28 band, France vs US/India per 1,000 S1.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import polars as pl
from ber import config
PB = 10_000_000
P = config.paths(); W = P["work"]; pq = P["parquet"] / "test"
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "name1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"), "ctry")
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "name1"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name")) for s in (2, 3)])
m = pl.read_csv(W / "output" / "v8u_s28_FIN" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
k = m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "e1", "matched_entity_ids": "eb"}).join(s1, on="e1").join(pool, on="eb")
raw = pl.read_parquet(W / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
k = k.join(raw, on=["q", "pid"], how="left")
fs = sorted(str(f) for f in (W / "features" / "test").glob("part_*.parquet"))
k = k.join(pl.scan_parquet(fs).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset").join(k.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect(), on=["q", "pid"], how="left")
ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
k = k.filter((ta.list.set_difference(tb).list.len() == 0) & (tb.list.set_difference(ta).list.len() >= 1) & (pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
k = k.with_columns(tb.list.set_difference(ta).list.sort().list.join("+").alias("added"),
                   pl.when(pl.col("p") < 0.9999).then(pl.lit("a<0.9999")).when(pl.col("p") < 0.99999).then(pl.lit("b0.9999-0.99999")).otherwise(pl.lit("c>=0.99999")).alias("band"))
n1 = {c: s1.filter(pl.col("ctry") == c).height for c in ("france", "us", "india")}
with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=40):
    for c in ("france", "us", "india"):
        x = k.filter(pl.col("ctry") == c)
        g = x.group_by("band").len().with_columns((1000 * pl.col("len") / n1[c]).round(2).alias("per1k")).sort("band")
        print(f"== {c}: words-added pairs at the exact address kept by FIN, by band"); print(g)
        print(x.filter(pl.col("band") == "b0.9999-0.99999").group_by("added").len().sort("len", descending=True).head(15))
    fr = k.filter((pl.col("ctry") == "france") & (pl.col("band") == "b0.9999-0.99999"))
    print(fr.sample(min(20, fr.height), seed=0).select("p", "a_name", "b_name"))
PY
