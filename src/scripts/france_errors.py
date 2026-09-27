"""What does a France recipe keep, drop and restore? Usage: python src/scripts/france_errors.py MODEL RECIPE_RUN
MODEL is the stack (its pair_p and threshold give the raw prediction), RECIPE_RUN a France-variant output folder
(WORK/output/<RECIPE_RUN>/matching_results.tsv). Prints France counts per class and raw samples of: S1 over the slot cap in the raw
prediction (some of their pairs must be wrong), non-exact pairs the recipe keeps, pairs it drops at high p, and pairs it restores."""

import json
import sys

import polars as pl

from ber import config, decision

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}


def main() -> None:
    name, run = sys.argv[1], sys.argv[2]
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "business_name", "business_address", "core1", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"),
        pl.col("core1").alias("c1"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "business_name", "business_address", "core1"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("business_name").alias("nb"),
        pl.col("business_address").alias("ab"), pl.col("core1").alias("cb")) for s in (2, 3)])
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    raw = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p").with_columns(pl.lit(True).alias("in_raw"))
    fin = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = fin.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    fin = fin.rename({"source1_entity_id": "e1", "matched_entity_ids": "eb"}).join(s1.select("e1", "q"), on="e1").join(pool.select("eb", "pid"), on="eb").select(
        "q", "pid", pl.lit(True).alias("in_fin"))
    d = raw.join(fin, on=["q", "pid"], how="full", coalesce=True).with_columns(pl.col("in_raw").fill_null(False), pl.col("in_fin").fill_null(False))
    d = d.join(pp.select("q", "pid", pl.col("p").alias("p_all")), on=["q", "pid"], how="left").with_columns(pl.coalesce("p", "p_all").alias("p")).drop("p_all")
    d = d.join(s1, on="q").filter(pl.col("ctry") == "france").join(pool, on="pid")
    d = d.with_columns((pl.col("c1") == pl.col("cb")).alias("exact"), (pl.col("pid") // PID_BASE).alias("src"),
                       pl.when(pl.col("in_raw") & pl.col("in_fin")).then(pl.lit("kept")).when(pl.col("in_raw")).then(pl.lit("dropped")).otherwise(pl.lit("restored")).alias("cls"))
    with pl.Config(tbl_rows=20, tbl_width_chars=200, fmt_str_lengths=60):
        print(d.group_by("cls", "exact").agg(pl.len(), pl.col("p").mean().alias("mean_p")).sort("cls", "exact"), flush=True)

        def show(x: pl.DataFrame, title: str, n: int) -> None:
            print(f"\n==== {title} ({x.height})", flush=True)
            for r in x.sample(n=min(n, x.height), seed=7).sort("q").iter_rows(named=True):
                print(f"  p {r['p']:.4f} {r['cls'][:4]} | {r['n1']} | {r['a1']}\n              -> {r['nb']} | {r['ab']}")

        over = d.filter(pl.col("in_raw")).group_by("q", "src").len().join(pl.DataFrame({"src": [2, 3], "cap": [5, 6]}, schema={"src": pl.Int64, "cap": pl.Int64}), on="src").filter(pl.col("len") > pl.col("cap"))
        print(f"\nS1-source groups over the cap in the raw prediction: {over.height}", flush=True)
        for q, src in over.select("q", "src").head(12).iter_rows():
            g = d.filter((pl.col("q") == q) & (pl.col("src") == src) & pl.col("in_raw")).sort("p", descending=True)
            r0 = g.row(0, named=True)
            print(f"\n  S1 {r0['n1']} | {r0['a1']}  (S{src})")
            for r in g.iter_rows(named=True):
                print(f"     p {r['p']:.4f} {r['cls'][:4]} | {r['nb']} | {r['ab']}")
        show(d.filter((pl.col("cls") == "kept") & ~pl.col("exact") & (pl.col("p") < 0.9999)), "KEPT non-exact, p < 0.9999", 45)
        show(d.filter((pl.col("cls") == "kept") & pl.col("exact") & (pl.col("p") < 0.99)), "KEPT exact, p < 0.99", 25)
        show(d.filter((pl.col("cls") == "dropped") & (pl.col("p") >= 0.95)), "DROPPED p >= 0.95", 45)
        show(d.filter((pl.col("cls") == "dropped") & (pl.col("p") < 0.95)), "DROPPED p < 0.95", 25)
        show(d.filter(pl.col("cls") == "restored"), "RESTORED", 25)


if __name__ == "__main__":
    main()
