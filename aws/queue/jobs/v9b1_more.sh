# v9 (CPU lane jobs2): where can v8u_s28_FIN (s29 version scored 0.987745) still drop French decoys? (a) its kept pairs by relation kind x address
# relation x raw s28 band ([0.72,0.9999), [0.9999,0.99999), >=0.99999) per 1,000 S1, France vs US vs India; (b) the same for the kept pairs the
# transfer-only model s28T rejects (its p < its threshold 0.70), plus examples.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import sys
import polars as pl
sys.path.insert(0, "src/scripts")
from ber import config, decision
from band_kinds import kinds
PB = 10_000_000
P = config.paths(); W = P["work"]; pq = P["parquet"] / "test"
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "name1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"), "ctry")
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "name1", "addr"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name"), (pl.col("addr") == "").alias("b_empty")) for s in (2, 3)])
m = pl.read_csv(W / "output" / "v8u_s28_FIN" / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
k = m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "e1", "matched_entity_ids": "eb"})
k = k.join(s1, on="e1").join(pool, on="eb")
raw = pl.read_parquet(W / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
t = pl.read_parquet(W / "output" / "s28T" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).select("q", "pid", pl.col("p").alias("pT"))
k = k.join(raw, on=["q", "pid"], how="left").join(t, on=["q", "pid"], how="left")
fs = sorted(str(f) for f in (W / "features" / "test").glob("part_*.parquet"))
k = k.join(pl.scan_parquet(fs).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset").join(k.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect(), on=["q", "pid"], how="left")
k = kinds(k).with_columns(pl.when(pl.col("a_core") == pl.col("b_core")).then(pl.lit("exact_core")).otherwise(pl.col("kind")).alias("kind"),
    pl.when(pl.col("b_empty")).then(pl.lit("empty")).when((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90)).then(pl.lit("exact_addr")).otherwise(pl.lit("other_addr")).alias("addr"),
    pl.when(pl.col("p") < 0.9999).then(pl.lit("a<0.9999")).when(pl.col("p") < 0.99999).then(pl.lit("b0.9999-0.99999")).otherwise(pl.lit("c>=0.99999")).alias("band"),
    (pl.col("pT") < 0.70).alias("T_rejects"))
n1 = {c: s1.filter(pl.col("ctry") == c).height for c in ("france", "us", "india")}
def rates(x, by):
    g = x.group_by("ctry", *by).len().pivot(on="ctry", index=by, values="len").fill_null(0)
    for c in ("france", "us", "india"):
        if c not in g.columns: g = g.with_columns(pl.lit(0).alias(c))
        g = g.with_columns((1000 * pl.col(c) / n1[c]).round(2).alias(c))
    return g.with_columns(((pl.col("france") - pl.max_horizontal("us", "india")) / pl.col("france")).round(2).alias("decoy_vs_max"),
                          ((pl.col("france") - pl.min_horizontal("us", "india")) / pl.col("france")).round(2).alias("decoy_vs_min"),
                          (pl.col("france") * n1["france"] / 1000).round(0).alias("fr_pairs"))
with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=40):
    print("(a) kept pairs by band x kind x address, per 1,000 S1 (bands b and c only, cells with >= 1,000 French pairs)")
    print(rates(k.filter(pl.col("band") != "a<0.9999"), ["band", "kind", "addr"]).filter(pl.col("fr_pairs") >= 1000).sort("decoy_vs_max", descending=True))
    print("(b) kept pairs the transfer-only model rejects, by kind x address")
    tr = k.filter(pl.col("T_rejects"))
    print(rates(tr, ["kind", "addr"]).filter(pl.col("fr_pairs") >= 300).sort("fr_pairs", descending=True))
    x = tr.filter(pl.col("ctry") == "france")
    print(f"France kept pairs T rejects: {x.height}; examples"); print(x.sample(min(25, x.height), seed=0).select("p", "pT", "kind", "addr", "a_name", "b_name"))
PY
