# v8 (CPU lane jobs2): France one-word swaps that typeswap does not cover (p >= 0.72, s28): swapped-in words by count, document frequency, mean p and
# slot ratio (rate at S1 with >= 3 exact copies in the source against none, as typeswap learns its 30 words), with examples; plus each word's
# pool-to-S1 frequency ratio per country (noise words the generator injects into copies sit far above 1; type words near 1).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
from difflib import SequenceMatcher
import numpy as np, polars as pl
from ber import config, decision
PB = 10_000_000
NOISE = ["fils", "groupe", "services", "developpement", "france", "associes", "cie", "and", "et"]
TYPE30 = ["college", "fetes", "compagnie", "culture", "parents", "primaire", "foyer", "maison", "section", "club", "jeunes", "amis", "societe", "union", "sportive",
          "ehpad", "centre", "loisirs", "l", "comite", "a", "amicale", "anciens", "sante", "service", "federation", "conseil", "maternelle", "ecole", "culturelle"]
P = config.paths(); pq = P["parquet"] / "test"
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), "ctry")
pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "ctry"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), pl.col("core1").alias("b_core"), pl.col("ctry").alias("pctry"), pl.lit(s).alias("src")) for s in (2, 3)])
# pool-to-S1 word frequency ratio per country
def wf(df, col, c):
    return df.filter(pl.col(c) == cc).select(pl.col(col).str.split(" ").list.unique().alias("w")).explode("w").group_by("w").len()
with pl.Config(tbl_rows=45, tbl_width_chars=200):
    for cc in ("france", "us", "india"):
        a = wf(s1, "a_core", "ctry").rename({"len": "n_s1"}); b = wf(pool, "b_core", "pctry").rename({"len": "n_pool"})
        n1, n2 = s1.filter(pl.col("ctry") == cc).height, pool.filter(pl.col("pctry") == cc).height
        r = a.join(b, on="w", how="full", coalesce=True).fill_null(0).filter(pl.col("n_pool") >= 300).with_columns(((pl.col("n_pool") / n2) / ((pl.col("n_s1") + 1) / n1)).alias("pool_s1_ratio"))
        print(f"== {cc}: words most over-represented in pool names against S1 names (>= 300 pool names)")
        print(r.sort("pool_s1_ratio", descending=True).head(25))
pp = pl.read_parquet(P["work"] / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
own = decision.assign_exclusive(pp).filter(pl.col("p") >= 0.72).join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left")
exs = own.filter((pl.col("a_core") == pl.col("b_core")) & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
slots = s1.filter(pl.col("ctry") == "france").select("q").join(pl.DataFrame({"src": [2, 3]}, schema={"src": pl.Int32}), how="cross").join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
nA, nB = slots.filter(pl.col("k") >= 3).height, slots.filter(pl.col("k") == 0).height
ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
sw = own.filter((ta.list.set_difference(tb).list.len() == 1) & (tb.list.set_difference(ta).list.len() == 1)).with_columns(
    ta.list.set_difference(tb).list.first().alias("xa"), tb.list.set_difference(ta).list.first().alias("xb"))
sw = sw.with_columns(pl.Series("sim", [SequenceMatcher(None, x, y).ratio() for x, y in zip(sw["xa"].to_list(), sw["xb"].to_list())]))
df1 = s1.filter(pl.col("ctry") == "france").select(pl.col("a_core").str.split(" ").list.unique().alias("w")).explode("w").group_by("w").len().rename({"len": "df"})
rest = sw.filter((pl.col("sim") < 0.7) & ~pl.col("xb").is_in(NOISE) & ~pl.col("xa").is_in(NOISE) & ~pl.col("xb").is_in(TYPE30)).join(slots, on=["q", "src"], how="left")
rest = rest.join(df1.rename({"w": "xb", "df": "df_xb"}), on="xb", how="left").join(df1.rename({"w": "xa", "df": "df_xa"}), on="xa", how="left").fill_null(0)
print(f"France one-word swaps (p >= 0.72): {sw.height}; typo-like {sw.filter(pl.col('sim') >= 0.7).height}; noise {sw.filter(pl.col('xb').is_in(NOISE) | pl.col('xa').is_in(NOISE)).height}; "
      f"into the 30 type words {sw.filter(pl.col('xb').is_in(TYPE30)).height}; rest {rest.height} (p >= 0.995: {rest.filter(pl.col('p') >= 0.995).height})")
g = rest.group_by("xb").agg(pl.len().alias("n"), pl.col("df_xb").first(), pl.col("p").mean().alias("mean_p"), (pl.col("k") >= 3).sum().alias("nA"), (pl.col("k") == 0).sum().alias("nB"),
                            pl.col("xa").value_counts(sort=True).head(3).struct.field("xa").alias("top_xa"))
g = g.with_columns(((pl.col("nA") / nA) / (pl.col("nB") / nB + 1e-12)).alias("ratio"))
with pl.Config(tbl_rows=70, tbl_width_chars=220, fmt_str_lengths=60, fmt_table_cell_list_len=3):
    print("swapped-in words of the rest (>= 15 pairs), by count:"); print(g.filter(pl.col("n") >= 15).sort("n", descending=True).head(70))
    print("rest by document frequency of both words (both >= 100 = two common words):")
    print(rest.with_columns(((pl.col("df_xa") >= 100) & (pl.col("df_xb") >= 100)).alias("both_common"), ((pl.col("df_xa") >= 20) & (pl.col("df_xb") >= 20)).alias("both_20")).group_by("both_common", "both_20")
          .agg(pl.len().alias("n"), pl.col("p").mean(), (pl.col("k") >= 3).sum().alias("nA"), (pl.col("k") == 0).sum().alias("nB")).with_columns(((pl.col("nA") / nA) / (pl.col("nB") / nB + 1e-12)).alias("ratio")))
    print(rest.filter(pl.col("p") >= 0.995).sample(40, seed=1).select("p", "src", "k", "a_core", "b_core", "df_xa", "df_xb"))
PY
