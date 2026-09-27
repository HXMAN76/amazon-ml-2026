"""Are France's extra one-match S1 singletons that caught a namesake? Usage: python src/scripts/single_match.py MODEL RECIPE_RUN
France has about 1% more S1 with exactly one predicted match than the US and India (single_pair.py). A singleton (no true match) scores 1 when
left empty and 0 with any match, so a namesake caught by a singleton costs a whole S1. Per country: the share of S1 by number of predicted
matches, and for S1 with a single match (or whose matches are all "away": another street or another house number) the cell of that match
(street same / mismatch / unknown, house same / diff, exact name, namesake count). The labelled holdout gives, per cell, the share of such S1
that are singletons and the share whose match is true (US, India); France uses RECIPE_RUN's final pairs, the others the model's raw pairs."""

import json
import sys

import polars as pl

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street

PID_BASE = 10_000_000


def cells(pred: pl.DataFrame, s1c: pl.DataFrame, poolc: pl.DataFrame, rare: pl.Series) -> pl.DataFrame:
    pc = poolc.join(pred.select("pid").unique(), on="pid", how="semi")
    pc = street(pc, "pid", "b_addr", "b_st", rare).select("pid", "b_core", "nb", "ab", "b_st", house(pl.col("b_addr")).alias("b_h"))
    d = pred.join(s1c, on="q").join(pc, on="pid")
    d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
    return d.with_columns(pl.when(pl.col("a_h").is_null() | pl.col("b_h").is_null()).then(pl.lit("unk")).when(pl.col("a_h") == pl.col("b_h")).then(pl.lit("same"))
                          .otherwise(pl.lit("diff")).alias("hs"), (pl.col("a_core") == pl.col("b_core")).alias("exact"),
                          pl.col("st").replace_strict({0: "same", 1: "MIS", 2: "unk"}, return_dtype=pl.Utf8).alias("street")).with_columns(
                          ((pl.col("st") == 1) | (pl.col("hs") == "diff")).alias("away"))


def main() -> None:
    name, run = sys.argv[1], sys.argv[2]
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
            from ber.split import holdout_q
            thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
            ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(ph).filter(pl.col("p") >= thr)
            lab = pl.read_parquet(pq / "labels.parquet").group_by("s1_rid").len().select(pl.col("s1_rid").cast(pl.Int64).alias("q"), pl.col("len").alias("n_true"))
            universe = pl.DataFrame({"q": holdout_q().astype("int64")}).join(lab, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
        else:
            thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
            pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            pred = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
            fin = pl.read_csv(P["work"] / "output" / run / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
            fin = fin.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            fin = fin.select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb"))
            fin = fin.join(s1.select("e1", "q"), on="e1").join(pool.select("eb", "pid"), on="eb").select("q", "pid").join(pp.select("q", "pid", "p"), on=["q", "pid"], how="left")
            universe = s1.select("q", pl.lit(None, dtype=pl.Int64).alias("n_true"))
        for c in s1["ctry"].unique().sort().to_list():
            s1c = s1.filter(pl.col("ctry") == c)
            poolc = pool.filter(pl.col("ctry_b") == c)
            rare = rare_vocab(s1c["a_addr"], poolc["b_addr"])
            s1c = street(s1c, "q", "a_addr", "a_st", rare).with_columns(house(pl.col("a_addr")).alias("a_h"), pl.len().over("a_core").alias("n_ns"))
            pr = fin if (split == "test" and c == "france") else pred
            d = cells(pr, s1c, poolc, rare)
            uni = universe.join(s1c.select("q"), on="q")
            per = uni.join(d.group_by("q").agg(pl.len().alias("n_pred"), pl.col("away").all().alias("all_away"), pl.col("label").sum().alias("tp") if "label" in d.columns else pl.lit(None).alias("tp")),
                           on="q", how="left").with_columns(pl.col("n_pred").fill_null(0))
            with pl.Config(tbl_rows=40, tbl_width_chars=220):
                print(f"\n== {split} {c}: {uni.height} S1, {d.height} predicted pairs", flush=True)
                print(per.group_by(pl.col("n_pred").clip(upper_bound=6)).agg(pl.len().alias("s1"), (pl.len() / per.height).alias("share"),
                      (pl.col("n_true") == 0).mean().alias("singleton_share")).sort("n_pred"), flush=True)
                one = d.join(per.filter(pl.col("n_pred") == 1).select("q", "n_true"), on="q")
                ns = pl.when(pl.col("n_ns") == 1).then(pl.lit("1")).when(pl.col("n_ns") <= 5).then(pl.lit("2-5")).otherwise(pl.lit("6+")).alias("ns")
                agg = [pl.len().alias("s1"), (pl.len() / uni.height).alias("share_of_all_s1"), pl.col("p").mean().alias("mean_p")]
                if split == "train":
                    agg += [(pl.col("n_true") == 0).mean().alias("singleton_share"), pl.col("label").mean().alias("pair_true")]
                print("single-match S1 by the cell of their match:")
                print(one.group_by("street", "hs", "exact", ns).agg(agg).sort("street", "hs", "exact", "ns"), flush=True)
                aw = per.filter(pl.col("all_away") & (pl.col("n_pred") >= 1))
                print(f"S1 whose matches are all away (another street or house number): {aw.height} ({aw.height / uni.height:.4f})"
                      + (f", singleton share {float((aw['n_true'] == 0).mean()):.3f}" if split == "train" else ""), flush=True)
                if split == "test" and c == "france":
                    x = one.filter(pl.col("away"))
                    print(f"\nsamples: France single-match S1 whose match is away ({x.height})")
                    for r in x.sample(n=min(25, x.height), seed=11).iter_rows(named=True):
                        print(f"  p {r['p'] if r['p'] is not None else float('nan'):.4f} ns {r['n_ns']} {r['street']}/{r['hs']} | {r['n1']} | {r['a1']}\n        -> {r['nb']} | {r['ab']}")
                    d.select("q", "pid", "p", "st", "hs", "exact", "n_ns", "away").write_parquet(P["work"] / "output" / run / "france_cells.parquet")


if __name__ == "__main__":
    main()
