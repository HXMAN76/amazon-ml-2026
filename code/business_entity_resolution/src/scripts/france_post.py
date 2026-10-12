"""France post-rules on top of a France-variant run. Usage: python src/scripts/france_post.py MODEL RUN_IN RUN_OUT --rules nsaway[:NMIN],nsaway_all[:NMIN],coined
Reads WORK/output/RUN_IN/{matching_results,candidate_pairs}.tsv, writes WORK/output/RUN_OUT/ (candidates copied unchanged).
  nsaway:N     drop France pairs whose core names are equal, whose streets do not match (namesake_street.py) and whose name is shared by at
               least N France S1 (default 6). Generic city-brand names ("bordeaux club sarl") have namesakes in the same city and small house
               numbers coincide: France has 8 times the US rate of such pairs (0.87% of predicted pairs against 0.11%), while unique names
               show no excess; on the US/India holdout such pairs are 99.5% true, so the rule is France-only.
  nsaway_all:N the same for non-exact names too.
  coined       restore raw-predicted France pairs (MODEL's pair_p at its threshold) that RUN_IN dropped when the pool name is a single coined word
               (not in the vocabulary of France's S1 names, 6+ letters, no digits) at the S1's street and house number: generator aliases
               (`Kelojax`, `Syndelta`) that are true copies in the training data. A source's slot cap (5 S2, 6 S3) is respected.
Prints what each rule changes."""

import argparse
import json
import shutil

import polars as pl

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("run_in")
    ap.add_argument("run_out")
    ap.add_argument("--rules", required=True)
    a = ap.parse_args()
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / a.model / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "business_name"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"),
        pl.col("ctry").alias("ctry_b"), pl.col("business_name").alias("nb")) for s in (2, 3)])
    src_dir = P["work"] / "output" / a.run_in
    m = pl.read_csv(src_dir / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    pairs = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb"))
              .join(s1.select("e1", "q"), on="e1").join(pool.select("eb", "pid"), on="eb"))
    n0 = pairs.height
    fr_s1 = s1.filter(pl.col("ctry") == "france")
    fr_pool = pool.filter(pl.col("ctry_b") == "france")
    rare = rare_vocab(fr_s1["a_addr"], fr_pool["b_addr"])
    fr_s1 = street(fr_s1, "q", "a_addr", "a_st", rare).with_columns(house(pl.col("a_addr")).alias("a_h"), pl.len().over("a_core").alias("n_ns"))

    def cells(x: pl.DataFrame) -> pl.DataFrame:
        pc = fr_pool.join(x.select("pid").unique(), on="pid", how="semi")
        pc = street(pc, "pid", "b_addr", "b_st", rare).select("pid", "b_core", "nb", "b_st", house(pl.col("b_addr")).alias("b_h"))
        d = x.join(fr_s1, on="q").join(pc, on="pid")
        return d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())),
                              (pl.col("a_h").is_not_null() & (pl.col("a_h") == pl.col("b_h"))).alias("h_same"), (pl.col("a_core") == pl.col("b_core")).alias("exact"))

    drop = pl.DataFrame(schema={"q": pl.Int64, "pid": pl.Int64})
    add = pl.DataFrame(schema={"q": pl.Int64, "pid": pl.Int64})
    for spec in a.rules.split(","):
        k = spec.split(":")
        if k[0] in ("nsaway", "nsaway_all"):
            nmin = int(k[1]) if len(k) > 1 else 6
            d = cells(pairs.select("q", "pid"))
            c = (pl.col("st") == 1) & (pl.col("n_ns") >= nmin)
            if k[0] == "nsaway":
                c = c & pl.col("exact")
            x = d.filter(c).select("q", "pid")
            print(f"rule {spec}: drops {x.height} France pairs ({x.height / max(d.height, 1):.4f} of France's kept pairs)", flush=True)
            drop = pl.concat([drop, x]).unique()
        elif k[0] == "coined":
            pp = pl.read_parquet(P["work"] / "output" / a.model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            raw = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
            gone = raw.join(fr_s1.select("q"), on="q", how="semi").join(pairs.select("q", "pid"), on=["q", "pid"], how="anti")
            vocab = pl.Series(fr_s1["a_core"].str.split(" ").explode().unique().drop_nulls())
            d = cells(gone.select("q", "pid", "p"))
            coined = (pl.col("b_core").str.contains(r"^[a-z]{6,}$") & ~pl.col("b_core").is_in(vocab.implode()) & (pl.col("st") == 0) & pl.col("h_same"))
            x = d.filter(coined)
            # respect the slot caps of the S1 after the drops
            kept = pairs.join(drop, on=["q", "pid"], how="anti").with_columns((pl.col("pid") // PID_BASE).alias("src")).group_by("q", "src").len().rename({"len": "n"})
            x = x.with_columns((pl.col("pid") // PID_BASE).alias("src")).join(kept, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
            x = x.sort("p", descending=True).with_columns(pl.int_range(pl.len()).over("q", "src").alias("_r"))
            x = x.filter(pl.col("n") + pl.col("_r") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int64))
            print(f"rule coined: restores {x.height} France pairs (of {gone.height} raw pairs the input run dropped); examples:", flush=True)
            for r in x.head(12).iter_rows(named=True):
                print(f"   p {r['p']:.3f} {r['nb']}")
            add = pl.concat([add, x.select("q", "pid")]).unique()
        else:
            raise SystemExit(f"unknown rule {spec}")
    out = pairs.select("q", "pid").join(drop, on=["q", "pid"], how="anti")
    out = pl.concat([out, add.join(out, on=["q", "pid"], how="anti")])
    print(f"pairs {n0} -> {out.height} (dropped {drop.height}, restored {add.height})", flush=True)
    lists = out.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
    res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
    dst = P["work"] / "output" / a.run_out
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e1, eb in res.select("e1", "eb").iter_rows():
            f.write(f"{e1}\t{eb}\n")
    shutil.copy(src_dir / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
