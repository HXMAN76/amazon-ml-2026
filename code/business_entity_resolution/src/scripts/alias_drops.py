"""Which raw France pairs does a recipe drop although the pool record names the S1 as its alias ("X Co formerly known as <S1 name>")?
Usage: python src/scripts/alias_drops.py MODEL RUN. Prints counts by p zone, how many S1 lose them, samples, and the same count for pairs the
recipe keeps, so an alias-restore rule can be sized."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def main() -> None:
    name, run = sys.argv[1], sys.argv[2]
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "name1", "ctry", "business_name"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"), "ctry", pl.col("business_name").alias("n1"))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "name1", "name2", "has_alias", "business_name"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name"),
        "name2", "has_alias", pl.col("business_name").alias("nb")) for s in (2, 3)])
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    raw = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
    m = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb")).join(s1.select("e1", "q"), on="e1")
             .join(pool.select("eb", "pid"), on="eb").select("q", "pid", pl.lit(True).alias("kept")))
    d = raw.join(s1, on="q").filter(pl.col("ctry") == "france").join(pool, on="pid").join(fin, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False))
    d = d.with_columns((pl.col("has_alias") & (pl.col("name2").str.contains(pl.col("a_core"), literal=True) | (pl.col("name2") == pl.col("a_name")))).alias("alias_is_s1"))
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(d.group_by("alias_is_s1", "kept").agg(pl.len(), pl.col("p").mean().alias("mean_p"), (pl.col("p") >= 0.999).mean().alias("p_ge_0999")).sort("alias_is_s1", "kept"))
        x = d.filter(pl.col("alias_is_s1") & ~pl.col("kept"))
        print(f"alias pairs dropped: {x.height} over {x['q'].n_unique()} S1; S1 left empty by the recipe among them: "
              f"{x.join(fin.select('q').unique(), on='q', how='anti')['q'].n_unique()}")
        print(x.select("p", "n1", "nb", "a_core", "b_core", "name2").sample(n=min(25, x.height), seed=1))
        y = d.filter(pl.col("has_alias") & ~pl.col("alias_is_s1") & ~pl.col("kept"))
        print(f"other alias-form pairs dropped: {y.height}")
        print(y.select("p", "n1", "nb", "name2").sample(n=min(15, y.height), seed=1))


if __name__ == "__main__":
    main()
