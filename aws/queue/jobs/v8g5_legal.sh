# v8 (CPU lane jobs2): legal-form conflicts (joined forms in the legal field plus spaced forms such as "s a s" / "e u r l" in the name) in the predicted
# pairs of v8u_s28_AR per country, against the labelled holdout (true share of kept pairs and of all candidates by legal relation); France slot ratio
# and examples. Is "same name, other legal form" a French decoy kind the US/India-trained model accepts?
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
from ber.split import holdout_q
PB = 10_000_000
P = config.paths()
SP = {"s a r l": "sarl", "s a s u": "sasu", "e u r l": "eurl", "s a s": "sas", "s c i": "sci", "s n c": "snc", "e i r l": "eirl", "e i": "ei", "s a": "sa"}
def legal_set(name: pl.Expr, legal: pl.Expr) -> pl.Expr:
    s = pl.concat_str([pl.lit(" "), name, pl.lit(" ")])
    found = [pl.when(s.str.contains(" " + k + " ")).then(pl.lit(v)) for k, v in SP.items()]
    base = legal.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return pl.concat_list([base, pl.concat_list(found).list.drop_nulls()]).list.unique().list.sort()
def rel(d):
    la, lb = pl.col("la"), pl.col("lb")
    return d.with_columns(pl.when((la.list.len() == 0) & (lb.list.len() == 0)).then(pl.lit("none")).when(la.list.len() == 0).then(pl.lit("pool_only"))
                            .when(lb.list.len() == 0).then(pl.lit("s1_only")).when(la.list.set_intersection(lb).list.len() > 0).then(pl.lit("same"))
                            .otherwise(pl.lit("conflict")).alias("legal_rel"))
def texts(split):
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "legal", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"), pl.col("core1").alias("a_core"), legal_set(pl.col("name1"), pl.col("legal")).alias("la"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "legal"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("name1").alias("b_name"), pl.col("core1").alias("b_core"), legal_set(pl.col("name1"), pl.col("legal")).alias("lb")) for s in (2, 3)])
    return s1, pool
with pl.Config(tbl_rows=40, tbl_width_chars=220, fmt_str_lengths=45):
    s1, pool = texts("train")
    m = P["work"] / "models" / "s28"; thr = json.loads((m / "holdout.json").read_text())["stack_threshold"]
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    h = pl.read_parquet(m / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    kept = decision.assign_exclusive(h).filter(pl.col("p") >= thr).select("q", "pid", pl.lit(True).alias("kept"))
    h = rel(h.join(kept, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False)).join(s1, on="q", how="left").join(pool, on="pid", how="left"))
    h = h.with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    print("holdout (US/India, labelled): candidates and kept pairs by legal relation and exact core; true share")
    print(h.group_by("legal_rel", "core_eq").agg(pl.len().alias("cands"), pl.col("label").mean().alias("cand_true"), pl.col("kept").sum().alias("kept"),
          pl.col("label").filter(pl.col("kept")).mean().alias("kept_true"), pl.col("p").filter(pl.col("label") == 0).mean().alias("p_false")).sort("legal_rel", "core_eq"))
    s1, pool = texts("test")
    pp = pl.read_parquet(P["work"] / "output" / "v8u_s28_AR" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = rel(decision.assign_exclusive(pp).filter(pl.col("p") >= 0.72).join(s1, on="q", how="left").join(pool, on="pid", how="left")).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    print("test v8u_s28_AR predicted pairs by country, legal relation, exact core (share of the country's pairs)")
    t = own.group_by("ctry", "legal_rel", "core_eq").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"))
    print(t.join(own.group_by("ctry").len().rename({"len": "tot"}), on="ctry").with_columns((pl.col("pairs") / pl.col("tot")).alias("share")).drop("tot").sort("ctry", "legal_rel", "core_eq"))
    fr = own.filter((pl.col("ctry") == "france") & (pl.col("legal_rel") == "conflict"))
    exs = own.filter((pl.col("ctry") == "france") & pl.col("core_eq") & (pl.col("p") >= 0.999)).with_columns((pl.col("pid") // PB).alias("src")).group_by("q", "src").len().rename({"len": "k"})
    fr = fr.with_columns((pl.col("pid") // PB).alias("src")).join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    fq = s1.filter(pl.col("ctry") == "france").select("q").join(pl.DataFrame({"src": [2, 3]}, schema={"src": pl.Int64}), how="cross").join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    nA, nB = fq.filter(pl.col("k") >= 3).height, fq.filter(pl.col("k") == 0).height
    print(f"France conflict pairs {fr.height}: rate at S1 with >= 3 exact copies in the source / rate at S1 with none = "
          f"{(fr.filter(pl.col('k') >= 3).height / nA) / max(1e-12, fr.filter(pl.col('k') == 0).height / nB):.3f} (decoys about 1, true copies well below)")
    print(fr.sample(min(30, fr.height), seed=0).select("p", "core_eq", "a_name", "b_name"))
    print("France conflict pairs by legal pair:")
    print(fr.with_columns(pl.col("la").list.join("+").alias("la_s"), pl.col("lb").list.join("+").alias("lb_s")).group_by("la_s", "lb_s").len().sort("len", descending=True).head(20))
PY
