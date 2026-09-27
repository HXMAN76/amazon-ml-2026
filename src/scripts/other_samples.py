"""What do France's confident non-exact, non-swap pairs look like where decoys concentrate? Usage: python src/scripts/other_samples.py MODEL
Slot occupancy: an S1 with k >= 3 confident exact-name copies in a source has few free slots, so its other pairs in that source are decoy-rich; at
k = 0 they are mostly true. For France's predicted pairs with p >= 0.999 whose names are not equal, not a one-word swap, not initials, not glued
and not equal after spaced legal forms: the kinds of difference (legal form, house number, extra or missing words, address similarity) in both
groups, and raw examples of each, to find the pattern that separates them."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import strip_spaced

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "legal", "addr", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"), pl.col("core1").alias("a_core"), pl.col("legal").alias("a_legal"), pl.col("addr").alias("a_addr"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1", "legal", "addr"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("name1").alias("b_name"), pl.col("core1").alias("b_core"), pl.col("legal").alias("b_legal"),
        pl.col("addr").alias("b_addr"), pl.lit(s).alias("src")) for s in (2, 3)])
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left")
    k = d.filter((pl.col("a_core") == pl.col("b_core")) & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    d = d.join(k, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    a, b = pl.col("a_core"), pl.col("b_core")
    ta, tb = a.str.split(" ").list.unique(), b.str.split(" ").list.unique()
    one_swap = (ta.list.set_difference(tb).list.len() == 1) & (tb.list.set_difference(ta).list.len() == 1)
    tiny = b.str.len_chars().is_between(2, 3) & ~b.str.contains(" ")
    other = (a != b) & ~one_swap & ~tiny & (strip_spaced(a) != strip_spaced(b)) & (a.str.replace_all(" ", "") != b.str.replace_all(" ", ""))
    d = d.filter(other & (pl.col("p") >= 0.999))
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset", "name_tset", "legal_conflict")
              .join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    d = d.join(feat, on=["q", "pid"], how="left").with_columns(
        pl.when(pl.col("k") >= 3).then(pl.lit("full (k>=3)")).when(pl.col("k") == 0).then(pl.lit("free (k=0)")).otherwise(pl.lit("mid")).alias("slots"),
        (ta.list.set_difference(tb).list.len()).alias("s1_words_missing"), (tb.list.set_difference(ta).list.len()).alias("extra_pool_words"),
        (pl.col("house_eq") > 0.5).alias("same_house"), (pl.col("legal_conflict") > 0.5).alias("legal_conflict"))
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=240, fmt_str_lengths=38):
        print(f"France confident 'other' pairs of {name}: {d.height}")
        print(d.group_by("slots").agg(pl.len(), pl.col("same_house").mean(), pl.col("legal_conflict").mean(), (pl.col("addr_tset") >= 90).mean().alias("addr90"),
                                      pl.col("name_tset").mean(), pl.col("s1_words_missing").mean(), pl.col("extra_pool_words").mean()).sort("slots"))
        for g in ("full (k>=3)", "free (k=0)"):
            x = d.filter(pl.col("slots") == g)
            print(f"\n{g}: {x.height} pairs; examples")
            print(x.sample(min(25, x.height), seed=4).select("p", "src", "k", "a_name", "b_name", "a_addr", "b_addr"))


if __name__ == "__main__":
    main()
