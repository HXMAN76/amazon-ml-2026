"""Same-name records in another street: a noisy copy or a namesake business? Usage: python src/scripts/street_cluster.py MODEL [RECIPE_RUN]
A predicted pair is a street mismatch when the two addresses share no rare token (document frequency below 0.3% of the country's S1 addresses,
typos allowed: best token similarity below 80). Two instruments that do not rely on the slot fit (which cannot see exact-name decoys, since
those are counted among the S1's "sure" copies):
  mate   another pool record with the same core name as the candidate that shares a rare street token with it: the candidate's street has a
         business of its own (a hidden namesake with several copies), while a noisy copy of the S1 is a lone record.
  k_same the S1's other predicted records in its own street (it already has copies at home).
The labelled holdout gives the true share of every cell for the US and India; the test gives the cell sizes per country. With RECIPE_RUN
the France cells are split by whether the recipe keeps the pair."""

import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def rare_vocab(addr: pl.Series) -> set[str]:
    n = addr.len()
    vc = addr.str.split(" ").list.unique().explode().value_counts()
    return set(vc.filter(pl.col("count") < 0.003 * n).to_series(0).to_list())


def rare_list(addr: pl.Expr, rare: pl.Series) -> pl.Expr:
    return addr.str.split(" ").list.unique().list.eval(pl.element().filter(pl.element().is_in(rare) & ~pl.element().str.contains(r"\d")))


def mismatch(ra: list, rb: list) -> np.ndarray:
    out = np.zeros(len(ra), dtype=bool)
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not x or not y:
            continue
        x, y = set(x), set(y)
        if x & y:
            continue
        out[i] = max(fuzz.ratio(u, v) for u in x for v in y) < 80
    return out


def analyse(d: pl.DataFrame, pool_c: pl.DataFrame, tag: str) -> pl.DataFrame:
    """d: predicted pairs of one country with a_rare, b_rare, a_core, b_core (and label on the holdout)."""
    m = mismatch(d["a_rare"].to_list(), d["b_rare"].to_list())
    d = d.with_columns(pl.Series("mis", m))
    # k_same: the S1's other predicted records that share a rare token with the S1's address
    same = ~pl.col("mis") & (pl.col("b_rare").list.len() > 0) & (pl.col("a_rare").list.set_intersection(pl.col("b_rare")).list.len() > 0)
    d = d.with_columns(same.alias("_same")).with_columns((pl.col("_same").cast(pl.Int32).sum().over("q") - pl.col("_same").cast(pl.Int32)).alias("k_same"))
    # mate: another pool record with the same core name sharing a rare street token with the candidate
    mis = d.filter(pl.col("mis")).select("q", "pid", "b_core", pl.col("b_rare").alias("t")).explode("t")
    idx = pool_c.select(pl.col("pid").alias("pid2"), pl.col("core1").alias("b_core"), pl.col("rare").alias("t")).explode("t").drop_nulls("t")
    idx = idx.join(mis.select("b_core", "t").unique(), on=["b_core", "t"], how="semi")
    mates = mis.join(idx, on=["b_core", "t"]).filter(pl.col("pid2") != pl.col("pid")).group_by("q", "pid").agg(pl.col("pid2").n_unique().alias("n_mate"))
    d = d.join(mates, on=["q", "pid"], how="left").with_columns(pl.col("n_mate").fill_null(0))
    d = d.with_columns((pl.col("a_core") == pl.col("b_core")).alias("exact"))
    agg = [pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p")]
    if "label" in d.columns:
        agg.append(pl.col("label").mean().alias("true_share"))
    extra = ["kept"] if "kept" in d.columns else []
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(f"\n== {tag}: {d.height} predicted pairs, street mismatch {int(m.sum())} ({m.mean():.4f})", flush=True)
        print(d.filter(pl.col("mis")).group_by(["exact", (pl.col("n_mate") > 0).alias("mate"), (pl.col("k_same") > 0).alias("home_copies")] + extra).agg(agg)
              .sort(["exact", "mate", "home_copies"] + extra), flush=True)
        print(d.filter(~pl.col("mis")).group_by("exact").agg(agg).sort("exact"), flush=True)
    return d


def main() -> None:
    name = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) > 2 else None
    P = config.paths()
    for split in ("train", "test"):
        pq = P["parquet"] / split
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "business_name", "business_address"]).select(
            pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), "ctry",
            pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
        pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "business_name", "business_address"]).select(
            (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), "core1", "addr", pl.col("ctry").alias("ctry_b"),
            pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
        if split == "train":
            thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
            ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(ph).filter(pl.col("p") >= thr)
        else:
            thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
            pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
        for c in s1["ctry"].unique().sort().to_list():
            rare = pl.Series(sorted(rare_vocab(s1.filter(pl.col("ctry") == c)["a_addr"])))
            s1c = s1.filter(pl.col("ctry") == c).with_columns(rare_list(pl.col("a_addr"), rare).alias("a_rare"))
            pool_c = pool.filter(pl.col("ctry_b") == c).with_columns(rare_list(pl.col("addr"), rare).alias("rare"))
            d = pred.join(s1c, on="q").join(pool_c.select("pid", "eb", pl.col("core1").alias("b_core"), pl.col("rare").alias("b_rare"), "nb", "ab"), on="pid")
            if run and split == "test" and c == "france":
                fin = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
                fin = fin.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                fin = fin.select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb"), pl.lit(True).alias("kept"))
                d = d.join(fin, on=["e1", "eb"], how="left").with_columns(pl.col("kept").fill_null(False))
            d = analyse(d, pool_c, f"{split} {c}")
            if split == "test" and c == "france":
                with pl.Config(tbl_rows=40, fmt_str_lengths=70, tbl_width_chars=250):
                    for mate in (True, False):
                        x = d.filter(pl.col("mis") & pl.col("exact") & ((pl.col("n_mate") > 0) == mate))
                        print(f"\nsamples: France exact-name street mismatch, mate={mate}")
                        for r in x.sample(n=min(15, x.height), seed=3).iter_rows(named=True):
                            print(f"  p {r['p']:.4f} mates {r['n_mate']} home {r['k_same']} | {r['n1']} | {r['a1']}\n        -> {r['nb']} | {r['ab']}")
                d.filter(pl.col("mis")).select("q", "pid", "p", "exact", "n_mate", "k_same").write_parquet(P["work"] / "output" / name / "france_street_mis.parquet")


if __name__ == "__main__":
    main()
