"""Restore France pairs a recipe dropped whose only name difference is a misspelt word. Usage: python src/scripts/typo_restore.py MODEL RUN_IN RUN_OUT
France's decoys swap in a different type word (club / ecole); the generator's copies misspell a word (coordination / curdination, amicale / aimlcae).
Restored: raw-predicted France pairs (MODEL's pair_p at its threshold, p >= 0.9) that RUN_IN dropped, where exactly one word differs on each side and
the two words are at least 75% similar (character ratio) but are not both known type words (sport / sportif), there is no French legal-form conflict,
and the addresses are not in another street or at another house number (namesake_street.py). Slot caps 5 S2 / 6 S3 kept.
Prints counts, a slot-fit sanity share and samples; writes RUN_OUT."""

import json
import shutil
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street
from pool_support_scan import slot_fit

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}
TYPE_WORDS = {"amicale", "amis", "anciens", "atelier", "auto", "cafe", "centre", "club", "college", "comite", "compagnie", "conseil", "culture", "culturelle",
              "danse", "ecole", "ehpad", "elementaire", "federation", "fetes", "foyer", "gestion", "groupement", "institut", "jeunes", "loisirs", "lycee", "maison",
              "maternelle", "medico", "musique", "parents", "patrimoine", "pharmacie", "primaire", "residence", "sante", "section", "service", "societe", "soins",
              "sport", "sportif", "sportive", "union", "hotel", "theatre", "association", "clinique", "cercle", "collectif", "familles", "ateliers", "services"}
FR_LEGAL = ["sarl", "sas", "sasu", "eurl", "sci", "snc", "ei", "eirl", "sa", "scop"]


def main() -> None:
    model, run_in, run_out = sys.argv[1:4]
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "legal", "business_name", "business_address"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), "ctry",
        pl.col("legal").alias("a_legal"), "business_name", "business_address")
    fr = s1.filter(pl.col("ctry") == "france")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "legal", "business_name", "business_address"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"),
        pl.col("ctry").alias("ctry_b"), pl.col("legal").alias("b_legal"), pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
    frp = pool.filter(pl.col("ctry_b") == "france")
    pp = pl.read_parquet(P["work"] / "output" / model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    raw = decision.assign_exclusive(pp.join(fr.select("q"), on="q", how="semi")).filter(pl.col("p") >= thr).select("q", "pid", "p")
    m = pl.read_csv(P["work"] / "output" / run_in / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb")).join(s1.select("e1", "q"), on="e1")
             .join(pool.select("eb", "pid"), on="eb").select("q", "pid"))
    gone = raw.filter(pl.col("p") >= 0.9).join(fin, on=["q", "pid"], how="anti").join(fr, on="q").join(frp, on="pid")
    gone = gone.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    gone = gone.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).alias("add"), pl.col("ta").list.set_difference(pl.col("tb")).alias("gone"))
    one = gone.filter((pl.col("add").list.len() == 1) & (pl.col("gone").list.len() == 1)).with_columns(
        pl.col("add").list.first().alias("wb"), pl.col("gone").list.first().alias("wa"))
    sim = np.array([fuzz.ratio(x, y) for x, y in zip(one["wa"].to_list(), one["wb"].to_list())], dtype=np.float32)
    both_type = np.array([(x in TYPE_WORDS and y in TYPE_WORDS) for x, y in zip(one["wa"].to_list(), one["wb"].to_list())])
    one = one.with_columns(pl.Series("wsim", sim), pl.Series("both_type", both_type)).filter((pl.col("wsim") >= 75) & ~pl.col("both_type") & (pl.col("wa").str.len_chars() >= 4))
    la = pl.col("a_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
    lb = pl.col("b_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
    one = one.filter(~((la.list.len() > 0) & (lb.list.len() > 0) & (la.list.set_intersection(lb).list.len() == 0)).fill_null(False))
    rare = rare_vocab(fr["a_addr"], frp["b_addr"])
    one = street(street(one, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
    one = one.with_columns(pl.Series("st", compare(one["a_st"].to_list(), one["b_st"].to_list())), house(pl.col("a_addr")).alias("a_h"), house(pl.col("b_addr")).alias("b_h"))
    hdiff = pl.col("a_h").is_not_null() & pl.col("b_h").is_not_null() & (pl.col("a_h") != pl.col("b_h"))
    one = one.filter((pl.col("st") != 1) & ~hdiff).with_columns((pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src"))
    kept_n = fin.join(fr.select("q"), on="q", how="semi").with_columns((pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src")).group_by("q", "src").len().rename({"len": "n"})
    one = one.join(kept_n, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("p", descending=True)
    one = one.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
    print(f"typo restores: {one.height} France pairs (of {gone.height} dropped raw pairs with p >= 0.9); by street: {one.group_by('st').len().sort('st').to_dicts()}", flush=True)
    # slot-fit sanity (k = confident exact copies kept by RUN_IN per S1 and source)
    k = fin.join(fr.select("q", "a_core"), on="q").join(frp.select("pid", "b_core"), on="pid").filter(pl.col("a_core") == pl.col("b_core")).with_columns(
        (pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src")).group_by("q", "src").len().rename({"len": "k"})
    base = fr.select("q").join(pl.DataFrame({"src": [2, 3]}, schema={"src": pl.Int32}), how="cross").join(k, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    for s, cap in CAP.items():
        per = base.filter(pl.col("src") == s).join(one.filter(pl.col("src") == s).group_by("q").len().rename({"len": "n"}), on="q", how="left").with_columns(pl.col("n").fill_null(0))
        print(f"  slot fit S{s}: decoy rate, true rate, decoy share = {slot_fit(per, cap)}", flush=True)
    for r in one.sample(n=min(30, one.height), seed=5).iter_rows(named=True):
        print(f"  p {r['p']:.4f} {r['wa']}->{r['wb']} ({r['wsim']:.0f}) | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    out = pl.concat([fin, one.select("q", "pid")])
    lists = out.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
    res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
    dst = P["work"] / "output" / run_out
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e1, eb in res.select("e1", "eb").iter_rows():
            f.write(f"{e1}\t{eb}\n")
    shutil.copy(P["work"] / "output" / run_in / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    n_fr = raw.height
    print(f"wrote {dst}: pairs {fin.height} -> {out.height}; expected France F0.5 if 90% true: {one.height / n_fr * 100 * (0.9 * 0.0022 - 0.1 * 0.0063):+.4f}", flush=True)


if __name__ == "__main__":
    main()
