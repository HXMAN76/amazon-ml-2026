"""Type-word swaps hidden inside glued names and domains, and multi-word type changes, that the token rules cannot see.
Usage: python src/scripts/france/glued_typeswap.py MODEL RUN_IN RUN_OUT [--glued]
France's decoys are siblings with another type word (`lille club sarl` against `lille ecole sarl`). The typeswap rules compare word sets, so a
sibling written as one token (`lilleecolesarl.com`, `lilleecole`) passes. A France pair is flagged when the S1 name has a type word, the pool name
is a single token (glued or domain) that contains none of the S1's type words (typos allowed: partial ratio < 80), contains another type word
(4+ letters) and contains the S1's other words (so it is a sibling, not an unrelated coined name). Also flagged: multi-word pool names that lose
every S1 type word and gain another type word (typechange), at any p. Prints counts by RUN_IN kept/dropped and p band, samples, and writes
RUN_OUT = RUN_IN without the multi-word type changes it kept (with --glued also the glued/domain flags; measured on s29: mostly brands
that contain a type word, i.e. true copies, so off by default)."""

import json
import shutil
import sys

import polars as pl
from rapidfuzz import fuzz

from ber import config, decision

PID_BASE = 10_000_000
TYPE = ["amicale", "amis", "anciens", "atelier", "ateliers", "cafe", "centre", "club", "college", "comite", "compagnie", "conseil", "culture", "culturelle",
        "danse", "ecole", "ehpad", "elementaire", "federation", "fetes", "foyer", "gestion", "groupement", "institut", "jeunes", "loisirs", "lycee", "maison",
        "maternelle", "medico", "musique", "parents", "patrimoine", "pharmacie", "primaire", "residence", "sante", "section", "service", "societe", "soins",
        "sport", "sportif", "sportive", "union", "hotel", "theatre", "association", "clinique", "cercle", "collectif", "federation", "hopital", "internat",
        "creche", "garderie", "mutuelle", "syndicat", "chorale", "fanfare", "orchestre", "bibliotheque", "piscine", "stade", "tennis", "football", "rugby",
        "judo", "karate", "yoga", "gymnastique", "petanque", "boulangerie", "restaurant", "brasserie", "garage", "cabinet", "agence", "atelier"]
TYPE = sorted(set(TYPE), key=len, reverse=True)


def s1_types(words: list[str]) -> list[str]:
    return [w for w in words if w in TYPE]


def contains(g: str, w: str) -> bool:
    return w in g or (len(w) >= 5 and fuzz.partial_ratio(w, g) >= 80)


def flag_glued(a_core: str, b_core: str) -> bool:
    words = [w for w in a_core.split() if len(w) >= 3]
    ts = s1_types(words)
    if not ts or " " in b_core or len(b_core) < 6:
        return False
    g = b_core
    if any(contains(g, t) for t in ts):
        return False
    other = [t for t in TYPE if t not in ts and len(t) >= 4 and t in g]
    if not other:
        return False
    rest = [w for w in words if w not in ts and len(w) >= 3]
    return all(contains(g, w) for w in rest) if rest else False


def flag_multi(a_core: str, b_core: str) -> bool:
    ta, tb = set(a_core.split()), set(b_core.split())
    ts = [w for w in ta if w in TYPE]
    if not ts or len(tb) < 2:
        return False
    gone = [t for t in ts if t not in tb]
    gained = [t for t in tb - ta if t in TYPE]
    return len(gone) == len(ts) and len(gained) >= 1 and len((ta - set(ts)) & tb) >= 1


def main() -> None:
    model, run_in, run_out = sys.argv[1:4]
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "ctry", "business_name", "business_address"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), "ctry", "business_name", "business_address")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "is_domain", "business_name", "business_address"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), "is_domain",
        pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
    fr = s1.filter(pl.col("ctry") == "france")
    pp = pl.read_parquet(P["work"] / "output" / model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    raw = decision.assign_exclusive(pp.join(fr.select("q"), on="q", how="semi")).filter(pl.col("p") >= thr).select("q", "pid", "p")
    m = pl.read_csv(P["work"] / "output" / run_in / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb")).join(s1.select("e1", "q"), on="e1")
             .join(pool.select("eb", "pid"), on="eb").select("q", "pid"))
    fin_fr = fin.join(fr.select("q"), on="q", how="semi").with_columns(pl.lit(True).alias("kept"))
    # all France pairs RUN_IN keeps (restored ones too) plus the raw pairs it dropped
    d = pl.concat([raw, fin_fr.join(raw, on=["q", "pid"], how="anti").join(pp, on=["q", "pid"], how="left").select("q", "pid", "p")])
    d = d.join(fin_fr, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False)).join(fr, on="q").join(pool, on="pid")
    ac, bc = d["a_core"].fill_null("").to_list(), d["b_core"].fill_null("").to_list()
    d = d.with_columns(pl.Series("glued_swap", [flag_glued(x, y) for x, y in zip(ac, bc)]), pl.Series("multi_swap", [flag_multi(x, y) for x, y in zip(ac, bc)]))
    pb = pl.when(pl.col("p") < 0.99).then(pl.lit("<0.99")).when(pl.col("p") < 0.9999).then(pl.lit("<0.9999")).otherwise(pl.lit(">=0.9999")).alias("pb")
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(d.filter(pl.col("glued_swap") | pl.col("multi_swap")).group_by("glued_swap", "multi_swap", "is_domain", "kept", pb).len().sort("glued_swap", "multi_swap", "is_domain", "kept", "pb"), flush=True)
    for nm, f in (("GLUED/DOMAIN swap kept", pl.col("glued_swap") & pl.col("kept")), ("MULTI-word type change kept", pl.col("multi_swap") & pl.col("kept"))):
        x = d.filter(f)
        print(f"\n{nm}: {x.height}")
        for r in x.sample(n=min(30, x.height), seed=6).iter_rows(named=True):
            print(f"  p {r['p'] if r['p'] is not None else float('nan'):.4f} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    use = (pl.col("multi_swap") | pl.col("glued_swap")) if "--glued" in sys.argv else pl.col("multi_swap")  # glued flags catch brands that contain a type word: off by default
    drop = d.filter(pl.col("kept") & use).select("q", "pid")
    out = fin.join(drop, on=["q", "pid"], how="anti")
    lists = out.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
    res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
    dst = P["work"] / "output" / run_out
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e1, eb in res.select("e1", "eb").iter_rows():
            f.write(f"{e1}\t{eb}\n")
    shutil.copy(P["work"] / "output" / run_in / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    print(f"wrote {dst}: pairs {fin.height} -> {out.height} (France drops {drop.height}, {drop.height / max(fin_fr.height, 1):.4f} of France's kept pairs)", flush=True)


if __name__ == "__main__":
    main()
