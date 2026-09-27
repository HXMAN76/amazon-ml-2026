"""Generic names in the same city: is a same-name record in another street (or at another house number) the S1's copy or a namesake's?
Usage: python src/scripts/france/namesake_street.py MODEL [RECIPE_RUN]
France names are "<city> <type word> <legal form>" (530 S1 are called `bordeaux club sarl`), so namesakes share the city, and house numbers
are small: the same name, city and house number with another street happens by chance, while US/India namesakes live in other cities.
Street words = address tokens rare among BOTH the country's S1 and pool addresses (drops cities, regions and departments). Per predicted pair:
  street   same / mismatch (no rare token in common, typos allowed) / unknown (a side has no rare token)
  house    same / different first house number (leading zeros ignored)
  n_ns     number of S1 of the country with the same core name
  other_s1 another S1 with the same core name whose street words overlap the pool record's (the record's likely true owner)
The labelled holdout gives true shares per cell (US, India); the test gives France's cell sizes against the US and India."""

import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision

PID_BASE = 10_000_000


def rare_vocab(*addrs: pl.Series) -> pl.Series:
    """Tokens that are rare (document frequency below 0.3%) in every one of the given address columns."""
    seen: set[str] = set()
    common: set[str] = set()
    for a in addrs:
        n = a.len()
        vc = a.str.split(" ").list.unique().explode().value_counts()
        seen |= set(vc.to_series(0).drop_nulls().to_list())
        common |= set(vc.filter(pl.col("count") >= 0.003 * n).to_series(0).drop_nulls().to_list())
    return pl.Series(sorted(t for t in seen - common if t and not any(ch.isdigit() for ch in t) and len(t) >= 3))


def street(df: pl.DataFrame, key: str, col: str, out: str, rare: pl.Series) -> pl.DataFrame:
    """Add column `out`: the rare tokens of address column `col` (explode, semi-join, regroup by `key`)."""
    rs = pl.DataFrame({"t": rare})
    t = df.select(key, pl.col(col).str.split(" ").list.unique().alias("t")).explode("t").join(rs, on="t", how="semi")
    g = t.group_by(key).agg(pl.col("t").alias(out))
    return df.join(g, on=key, how="left").with_columns(pl.col(out).fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))


def house(addr: pl.Expr) -> pl.Expr:
    return addr.str.extract(r"(?:^|\s)0*(\d+)", 1)


def compare(ra: list, rb: list) -> np.ndarray:
    """0 same street, 1 mismatch, 2 unknown."""
    out = np.full(len(ra), 2, dtype=np.int8)
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not x or not y:
            continue
        x, y = set(x), set(y)
        if x & y or max(fuzz.ratio(u, v) for u in x for v in y) >= 80:
            out[i] = 0
        else:
            out[i] = 1
    return out


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
            (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"),
            pl.col("ctry").alias("ctry_b"), pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
        if split == "train":
            thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
            ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(ph).filter(pl.col("p") >= thr)
        else:
            thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
            pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
        for c in s1["ctry"].unique().sort().to_list():
            s1c = s1.filter(pl.col("ctry") == c)
            poolc = pool.filter(pl.col("ctry_b") == c)
            rare = rare_vocab(s1c["a_addr"], poolc["b_addr"])
            s1c = street(s1c, "q", "a_addr", "a_st", rare).with_columns(house(pl.col("a_addr")).alias("a_h"), pl.len().over("a_core").alias("n_ns"))
            pc = poolc.join(pred.select("pid").unique(), on="pid", how="semi")
            pc = street(pc, "pid", "b_addr", "b_st", rare).select("pid", "eb", "b_core", "nb", "ab", "b_st", house(pl.col("b_addr")).alias("b_h"))
            d = pred.join(s1c, on="q").join(pc, on="pid")
            d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
            d = d.with_columns(pl.when(pl.col("a_h").is_null() | pl.col("b_h").is_null()).then(pl.lit("unk")).when(pl.col("a_h") == pl.col("b_h")).then(pl.lit("same"))
                               .otherwise(pl.lit("diff")).alias("hs"), (pl.col("a_core") == pl.col("b_core")).alias("exact"))
            # another S1 namesake at the pool record's street (exact rare token shared)
            idx = s1c.filter(pl.col("n_ns") >= 2).select(pl.col("q").alias("y"), "a_core", pl.col("a_st").alias("t")).explode("t").drop_nulls("t")
            cand = d.filter((pl.col("st") == 1) & (pl.col("n_ns") >= 2)).select("q", "pid", "a_core", pl.col("b_st").alias("t")).explode("t").drop_nulls("t")
            oth = cand.join(idx, on=["a_core", "t"]).filter(pl.col("y") != pl.col("q")).group_by("q", "pid").agg(pl.col("y").n_unique().alias("n_other"))
            d = d.join(oth, on=["q", "pid"], how="left").with_columns(pl.col("n_other").fill_null(0))
            if run and split == "test" and c == "france":
                fin = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
                fin = fin.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                fin = fin.select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb"), pl.lit(True).alias("kept"))
                d = d.join(fin, on=["e1", "eb"], how="left").with_columns(pl.col("kept").fill_null(False))
            ns = pl.when(pl.col("n_ns") == 1).then(pl.lit("1")).when(pl.col("n_ns") <= 5).then(pl.lit("2-5")).otherwise(pl.lit("6+")).alias("ns")
            stl = pl.col("st").replace_strict({0: "same", 1: "MIS", 2: "unk"}, return_dtype=pl.Utf8).alias("street")
            agg = [pl.len().alias("pairs"), (pl.len() / d.height).alias("share"), pl.col("p").mean().alias("mean_p")]
            if "label" in d.columns:
                agg.append(pl.col("label").mean().alias("true_share"))
            if "kept" in d.columns:
                agg.append(pl.col("kept").mean().alias("kept_share"))
            with pl.Config(tbl_rows=60, tbl_width_chars=220):
                print(f"\n== {split} {c}: {d.height} predicted pairs", flush=True)
                print(d.group_by(stl, "hs", "exact", ns).agg(agg).filter(pl.col("street") != "same").sort("street", "hs", "exact", "ns"), flush=True)
                print(d.filter(pl.col("st") == 1).group_by("exact", ns, (pl.col("n_other") > 0).alias("other_s1_at_street")).agg(agg).sort("exact", "ns", "other_s1_at_street"), flush=True)
                if split == "test" and c == "france":
                    for title, f in (("MIS exact, 2+ namesakes, other S1 at the street", (pl.col("st") == 1) & pl.col("exact") & (pl.col("n_ns") >= 2) & (pl.col("n_other") > 0)),
                                     ("MIS exact, 2+ namesakes, no other S1 there", (pl.col("st") == 1) & pl.col("exact") & (pl.col("n_ns") >= 2) & (pl.col("n_other") == 0)),
                                     ("MIS exact, unique name", (pl.col("st") == 1) & pl.col("exact") & (pl.col("n_ns") == 1)),
                                     ("same street, other house number, exact", (pl.col("st") == 0) & (pl.col("hs") == "diff") & pl.col("exact"))):
                        x = d.filter(f)
                        print(f"\nsamples: {title} ({x.height})")
                        for r in x.sample(n=min(12, x.height), seed=5).iter_rows(named=True):
                            print(f"  p {r['p']:.4f} ns {r['n_ns']} other {r['n_other']} | {r['n1']} | {r['a1']}\n        -> {r['nb']} | {r['ab']}")
                    d.select("q", "pid", "p", "st", "hs", "exact", "n_ns", "n_other").write_parquet(P["work"] / "output" / name / "france_namesake_street.parquet")


if __name__ == "__main__":
    main()
