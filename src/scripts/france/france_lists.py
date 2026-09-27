"""Pair lists for france_variants.py's droplist / addlist rules, chosen by comparing France's rate per 1,000 S1 with the US and India rates
(the labelled holdout keeps these kinds 99%+ true, so a French excess over the US/India rate is decoys). Usage: python src/scripts/france/france_lists.py RUN [OUTDIR [MODEL]]
RUN is a France-variant run (e.g. v8u_s28_ALL2) whose kept pairs the lists refer to; writes WORK/v8x/{fb_ns_ref, fb_nsnear_ref, fb_coined_hi}.parquet.
  fb_ns_ref      drop: exact core name on another street, the name shared by 11+ France S1 and raw p < 0.9999, or by 6+ and raw p < 0.99
                 (France 58 and 44 per 1,000 S1 against at most 5 for the US / India; no excess at p >= 0.9999 or for rarer names).
  fb_nsnear_ref  drop: the same for near-exact names (at most two words differ, one shared).
  fb_coined_hi   add back: plain coined pool names (one 6+ letter word outside France's S1 vocabulary, no alias marker) at the S1's street and
                 house number with raw p in [0.995, 0.9999) that RUN dropped: France 34 per 1,000 S1 against 34 (US), so no excess there
                 (below 0.995 France has 20 against 4-6: decoys, not added)."""

import json
import sys

import polars as pl

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street

PID_BASE = 10_000_000


def main() -> None:
    run = sys.argv[1]
    P = config.paths()
    pq = P["parquet"] / "test"
    out = P["work"] / (sys.argv[2] if len(sys.argv) > 2 else "v8x")
    out.mkdir(parents=True, exist_ok=True)
    model = sys.argv[3] if len(sys.argv) > 3 else "s28"  # the stacked model whose raw probabilities RUN was decoded from
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    raw = pl.read_parquet(P["work"] / "output" / model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    pa = pl.read_parquet(P["work"] / "output" / run / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    kept = decision.assign_exclusive(pa).filter(pl.col("p") >= thr).select("q", "pid")
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).filter(pl.col("ctry") == "france").select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr")).with_columns(pl.len().over("a_core").alias("n_ns"))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "ctry", "has_alias"]).filter(pl.col("ctry") == "france").select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"), "has_alias") for s in (2, 3)])
    rare = rare_vocab(s1["a_addr"], pool["b_addr"])

    def streets(d: pl.DataFrame) -> pl.DataFrame:
        d = street(street(d, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
        return d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))

    # namesakes on another street, among RUN's kept pairs
    d = kept.join(raw.select("q", "pid", pl.col("p").alias("p_raw")), on=["q", "pid"], how="left").join(s1, on="q").join(pool, on="pid").filter(pl.col("n_ns") >= 6)
    ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
    d = d.with_columns((pl.col("a_core") == pl.col("b_core")).alias("exact"),
                       ((ta.list.set_difference(tb).list.len() + tb.list.set_difference(ta).list.len() <= 2) & (ta.list.set_intersection(tb).list.len() >= 1)
                        & (pl.col("a_core") != pl.col("b_core"))).alias("near"))
    d = streets(d.filter(pl.col("exact") | pl.col("near")))
    sel = (pl.col("st") == 1) & (((pl.col("n_ns") >= 11) & (pl.col("p_raw") < 0.9999)) | (pl.col("p_raw") < 0.99))
    for name, f in (("fb_ns_ref", pl.col("exact")), ("fb_nsnear_ref", pl.col("near"))):
        x = d.filter(sel & f).select("q", "pid")
        x.write_parquet(out / f"{name}.parquet")
        print(f"{name}: {x.height} pairs", flush=True)

    # plain coined names at the S1's street and house number, raw p in [0.995, 0.9999), dropped by RUN
    vocab = s1["a_core"].str.split(" ").explode().unique().drop_nulls().implode()
    c = (decision.assign_exclusive(raw).filter((pl.col("p") >= 0.995) & (pl.col("p") < 0.9999)).select("q", "pid").join(kept, on=["q", "pid"], how="anti")
         .join(s1, on="q").join(pool, on="pid").filter(pl.col("b_core").str.contains(r"^[a-z]{6,}$") & ~pl.col("b_core").is_in(vocab) & ~pl.col("has_alias")))
    c = streets(c).with_columns(house(pl.col("a_addr")).alias("a_h"), house(pl.col("b_addr")).alias("b_h"))
    x = c.filter((pl.col("st") == 0) & pl.col("a_h").is_not_null() & (pl.col("a_h") == pl.col("b_h"))).select("q", "pid")
    x.write_parquet(out / "fb_coined_hi.parquet")
    print(f"fb_coined_hi: {x.height} pairs", flush=True)


if __name__ == "__main__":
    main()
