"""How does the generator fill a shared building? Usage: python src/scripts/france_explore.py MODEL
Groups S1 and pool records by a loose address key (first house number + longest street word) and compares, per number of S1
in the group: pool records per S1 in the group, how many are predicted (test) or truly owned (train), and how many are left.
Then prints raw samples of France groups (test, with the model's owner and p) and of train groups (with the true owner),
so the decoy mechanism can be read off real records."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000


def key(col: str) -> pl.Expr:
    num = pl.col(col).str.extract(r"(\d+)", 1)
    word = pl.col(col).str.extract_all(r"[a-z]{4,}").list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True).first()).list.first()
    return pl.when(num.is_not_null() & word.is_not_null()).then(num + "|" + word).otherwise(None).alias("k")


def load(split: str):
    pq = config.paths()["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "business_name", "business_address", "addr", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"), "addr", "ctry").with_columns(key("addr")).drop("addr")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "business_name", "business_address", "addr", "ctry"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab"), "addr",
        pl.col("ctry").alias("ctry_b")) for s in (2, 3)]).with_columns(key("addr")).drop("addr")
    return s1, pool


def bucket(n: pl.Expr) -> pl.Expr:
    return pl.when(n == 1).then(pl.lit("1")).when(n == 2).then(pl.lit("2")).when(n <= 5).then(pl.lit("3-5")).when(n <= 14).then(pl.lit("6-14")).otherwise(pl.lit("15+")).alias("bucket")


def group_table(s1: pl.DataFrame, pool: pl.DataFrame, own: pl.DataFrame, ctry: str, tag: str) -> None:
    """own: (pid, q) owner of each pool record (truth or prediction)."""
    a = s1.filter((pl.col("ctry") == ctry) & pl.col("k").is_not_null())
    g = a.group_by("k").agg(pl.len().alias("n_s1"), pl.col("q"))
    pk = pool.filter((pl.col("ctry_b") == ctry) & pl.col("k").is_not_null()).join(own, on="pid", how="left")
    pk = pk.join(a.select("q", pl.col("k").alias("k_owner")), on="q", how="left")
    per = pk.group_by("k").agg(pl.len().alias("n_pool"), (pl.col("k_owner") == pl.col("k")).sum().alias("n_owned_here"),
                               pl.col("q").is_null().sum().alias("n_unowned"), (pl.col("q").is_not_null() & (pl.col("k_owner") != pl.col("k"))).sum().alias("n_owned_elsewhere"))
    t = g.join(per, on="k", how="left").fill_null(0).with_columns(bucket(pl.col("n_s1")))
    owned_total = own.join(a.select("q", "k"), on="q").group_by("k").len().rename({"len": "n_matches_of_group"})
    t = t.join(owned_total, on="k", how="left").fill_null(0)
    r = t.group_by("bucket").agg(pl.len().alias("groups"), pl.col("n_s1").sum().alias("s1"),
                                 (pl.col("n_pool").sum() / pl.col("n_s1").sum()).alias("pool_here_per_s1"),
                                 (pl.col("n_owned_here").sum() / pl.col("n_s1").sum()).alias("owned_here_per_s1"),
                                 (pl.col("n_unowned").sum() / pl.col("n_s1").sum()).alias("unowned_here_per_s1"),
                                 (pl.col("n_owned_elsewhere").sum() / pl.col("n_s1").sum()).alias("owned_elsewhere_per_s1"),
                                 (pl.col("n_matches_of_group").sum() / pl.col("n_s1").sum()).alias("matches_per_s1")).sort("bucket")
    with pl.Config(tbl_rows=10, tbl_width_chars=220):
        print(f"\n== {tag} {ctry}\n{r}", flush=True)


def samples(s1: pl.DataFrame, pool: pl.DataFrame, own: pl.DataFrame, ctry: str, tag: str, lo: int, hi: int, n: int, seed: int) -> None:
    a = s1.filter((pl.col("ctry") == ctry) & pl.col("k").is_not_null())
    g = a.group_by("k").len().filter((pl.col("len") >= lo) & (pl.col("len") <= hi)).sample(n=n, seed=seed)
    names = a.select("q", pl.col("n1").alias("owner_name"))
    pk = pool.filter(pl.col("ctry_b") == ctry).join(own, on="pid", how="left").join(names, on="q", how="left")
    for k in g["k"].to_list():
        print(f"\n---- {tag} {ctry} key {k}", flush=True)
        for r in a.filter(pl.col("k") == k).sort("n1").iter_rows(named=True):
            print(f"  S1 {r['q']:>9} | {r['n1']} | {r['a1']}")
        rows = pk.filter(pl.col("k") == k).sort("owner_name", "nb", nulls_last=True)
        for r in rows.head(60).iter_rows(named=True):
            p = f"{r['p']:.4f}" if r.get("p") is not None else "  -   "
            mark = "*" if r.get("pred", True) and r["owner_name"] is not None else " "
            print(f"   {mark}pool {r['pid']:>9} p {p} | {r['nb']} | {r['ab']} -> {r['owner_name']}")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s29"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1, pool = load("test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    best = decision.assign_exclusive(pp)
    pred = best.filter(pl.col("p") >= thr).select("pid", "q", "p")
    for c in ("france", "us", "india"):
        group_table(s1, pool, pred.select("pid", "q"), c, f"TEST predicted ({name}, thr {thr:.3f})")
    # train truth
    s1r, poolr = load("train")
    lab = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").select((pl.col("other_rid").cast(pl.Int64) + pl.col("src").cast(pl.Int64) * PID_BASE).alias("pid"),
                                                                          pl.col("s1_rid").cast(pl.Int64).alias("q"))
    for c in ("us", "india"):
        group_table(s1r, poolr, lab, c, "TRAIN truth")
    # raw samples: France test groups with the model's owner (p of the best S1, also below threshold)
    samples(s1, pool, best.select("pid", "q", "p").with_columns((pl.col("p") >= thr).alias("pred")), "france", "TEST", 3, 8, 8, 1)
    samples(s1r, poolr, lab.with_columns(pl.lit(None, dtype=pl.Float64).alias("p")), "us", "TRAIN", 3, 8, 5, 1)
    samples(s1r, poolr, lab.with_columns(pl.lit(None, dtype=pl.Float64).alias("p")), "india", "TRAIN", 3, 8, 3, 1)


if __name__ == "__main__":
    main()
