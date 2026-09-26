"""Pairs with the same name and house number but a different street ("19 rue emile gery bordeaux" against "bordeaux 19 rue du fort louis"): how many, per country?
Usage: python src/scripts/street_mismatch.py MODEL. Rare tokens = address tokens with document frequency below 0.3% of the split's S1 addresses (street names, not cities, regions,
departments or street types). A pair is a street mismatch when its two addresses share no rare token, even allowing typos (best token pair similarity below 80). The
labelled holdout gives the rate among true predicted pairs (the template); France and the US test are compared against it, also by number of exact copies k of the S1."""

import json
import re
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000


def rare_sets(addr: list[str], rare: set[str]) -> list[set[str]]:
    return [{t for t in a.split() if t in rare and not t.isdigit()} for a in addr]


def mismatch(a_addr: list[str], b_addr: list[str], rare: set[str]) -> np.ndarray:
    ra, rb = rare_sets(a_addr, rare), rare_sets(b_addr, rare)
    out = np.zeros(len(a_addr), dtype=bool)
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not x or not y:
            continue
        if x & y:
            continue
        out[i] = max(fuzz.ratio(u, v) for u in x for v in y) < 80
    return out


def rare_vocab(addr: pl.Series) -> set[str]:
    n = addr.len()
    df = addr.str.split(" ").list.unique().explode().value_counts()
    return set(df.filter(pl.col("count") < 0.003 * n)["addr"].to_list()) if "addr" in df.columns else set(df.filter(pl.col("count") < 0.003 * n).to_series(0).to_list())


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    res = {}
    for split in ("train", "test"):
        pq = P["parquet"] / split
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
        pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr"})
        # rare tokens per country: document frequency inside the country's S1 addresses
        for c in s1["ctry"].unique().to_list():
            res[(split, c)] = rare_vocab(s1.filter(pl.col("ctry") == c)["a_addr"])
        if split == "train":
            hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
            ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            d = decision.assign_exclusive(ph).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
        else:
            pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
        feat = (pl.scan_parquet(sorted(str(f) for dd in (["train", "train_rest"] if split == "train" else ["test"]) for f in (P["work"] / "features" / dd).glob("part_*.parquet")))
                  .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "house_eq").join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
        d = d.join(feat, on=["q", "pid"], how="left").filter((pl.col("name_tset") >= 95) & (pl.col("house_eq") > 0.5) & (pl.col("p") < 0.9999))
        ex = d.filter((pl.col("a_core") == pl.col("b_core"))).select("q", "pid")
        for c in d["ctry"].unique().to_list():
            x = d.filter(pl.col("ctry") == c)
            m = mismatch(x["a_addr"].to_list(), x["b_addr"].to_list(), res[(split, c)])
            x = x.with_columns(pl.Series("street_mismatch", m))
            lab = f", true share of the mismatching pairs {float(x.filter(pl.col('street_mismatch'))['label'].mean()):.3f}, of the others {float(x.filter(~pl.col('street_mismatch'))['label'].mean()):.4f}" if "label" in x.columns else ""
            print(f"{split} {c}: {x.height} pairs (same name >= 95, same house number, p < 0.9999); street mismatch {int(m.sum())} ({m.mean():.4f}){lab}")
            if split == "test" and c == "france":
                with pl.Config(tbl_rows=30, fmt_str_lengths=60, tbl_width_chars=220):
                    print(x.filter(pl.col("street_mismatch")).sample(min(30, int(m.sum())), seed=2).select("p", "a_addr", "b_addr"))
                print("by p zone:", x.group_by((pl.col("p") < 0.99).alias("p<0.99")).agg(pl.len().alias("pairs"), pl.col("street_mismatch").mean().alias("mismatch_rate")).to_dicts())


if __name__ == "__main__":
    main()
