"""Per-cell France excess over the US/India rate, to decide which pairs a France recipe should restore or drop.
Usage: python src/scripts/excess_cells.py MODEL RUN_IN RUN_OUT [--min-n 150] [--restore-share 0.15] [--drop-share 0.5]
A cell = name relation (exact, glued, initials, domain, alias, coined, reorder, words dropped, word added, noise added, one word swapped,
type-word change, typo, other) x address relation (same street + same house, same street other house, street mismatch, unknown) x p band.
On the labelled holdout these cells are 99%+ true, so France's pairs per 1,000 S1 above the US/India rate are decoys:
decoy share(cell) = max(0, rate_FR - ref) / rate_FR; restores use ref = min(rate_US, rate_IN) (the larger, safer share), drops use ref = max(...).
RUN_IN's France decisions are then revised: raw pairs RUN_IN dropped are restored in cells with a decoy share below --restore-share (never
type-word changes, French legal-form conflicts or namesakes in another street); pairs RUN_IN kept are dropped in cells with a share above
--drop-share. Prints the cell table, the expected France F0.5 and overall change (per 1% of France's pairs: +0.0022 for a true pair added,
-0.0063 for a wrong one; dropping mirrors it), and writes RUN_OUT (slot caps kept for restores)."""

import argparse
import json
import shutil

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}
TYPE_WORDS = ["amicale", "amis", "anciens", "atelier", "auto", "cafe", "centre", "club", "college", "comite", "compagnie", "conseil", "culture", "culturelle",
              "danse", "ecole", "ehpad", "elementaire", "federation", "fetes", "foyer", "gestion", "groupement", "institut", "jeunes", "loisirs", "lycee", "maison",
              "maternelle", "medico", "musique", "parents", "patrimoine", "pharmacie", "primaire", "residence", "sante", "section", "service", "societe", "soins",
              "sport", "sportif", "sportive", "union"]
NOISE = ["fils", "groupe", "services", "developpement", "france", "associes", "cie", "group", "center", "centre", "and", "co", "company", "holdings",
         "international", "enterprises", "solutions", "industries", "global", "trading", "corporation", "systems"]
FR_LEGAL = ["sarl", "sas", "sasu", "eurl", "sci", "snc", "ei", "eirl", "sa", "scop"]


def band(p: pl.Expr) -> pl.Expr:
    return (pl.when(p < 0.9).then(pl.lit("a<0.9")).when(p < 0.99).then(pl.lit("b<0.99")).when(p < 0.995).then(pl.lit("c<0.995"))
              .when(p < 0.9999).then(pl.lit("d<0.9999")).otherwise(pl.lit("e>=0.9999")))


def kinds(d: pl.DataFrame, vocab: pl.Series) -> pl.DataFrame:
    an, bn = d["a_core"].to_list(), d["b_core"].to_list()
    d = d.with_columns(pl.Series("nr", process.cpdist(an, bn, scorer=fuzz.ratio, dtype=np.float32, workers=-1)))
    d = d.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).alias("add"), pl.col("ta").list.set_difference(pl.col("tb")).alias("gone"))
    tw, nw = pl.Series(TYPE_WORDS).implode(), pl.Series(NOISE).implode()
    ini = pl.col("a_core").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    kind = (pl.when(pl.col("a_core") == pl.col("b_core")).then(pl.lit("exact"))
              .when(pl.col("a_core").str.replace_all(" ", "") == pl.col("b_core").str.replace_all(" ", "")).then(pl.lit("glued"))
              .when(pl.col("is_domain")).then(pl.lit("domain"))
              .when(pl.col("has_alias")).then(pl.lit("alias"))
              .when((pl.col("b_core").str.len_chars() <= 4) & ~pl.col("b_core").str.contains(" ") & (pl.col("b_core") == ini)).then(pl.lit("initials"))
              .when(pl.col("b_core").str.contains(r"^[a-z]{6,}$") & ~pl.col("b_core").is_in(vocab.implode())).then(pl.lit("coined"))
              .when((pl.col("add").list.len() == 0) & (pl.col("gone").list.len() == 0)).then(pl.lit("reorder"))
              .when(pl.col("add").list.eval(pl.element().is_in(tw)).list.any() & pl.col("gone").list.eval(pl.element().is_in(tw)).list.any()).then(pl.lit("typechange"))
              .when((pl.col("add").list.len() == 0)).then(pl.lit("dropped_words"))
              .when((pl.col("gone").list.len() == 0) & pl.col("add").list.eval(pl.element().is_in(nw)).list.all()).then(pl.lit("noise_added"))
              .when(pl.col("gone").list.len() == 0).then(pl.lit("word_added"))
              .when((pl.col("add").list.len() == 1) & (pl.col("gone").list.len() == 1) & pl.col("add").list.eval(pl.element().is_in(nw)).list.all()).then(pl.lit("noise_swap"))
              .when((pl.col("add").list.len() == 1) & (pl.col("gone").list.len() == 1)).then(pl.lit("swap"))
              .when(pl.col("nr") >= 85).then(pl.lit("typo"))
              .otherwise(pl.lit("other")))
    return d.with_columns(kind.alias("kind")).drop("ta", "tb", "add", "gone")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("run_in")
    ap.add_argument("run_out")
    ap.add_argument("--min-n", type=int, default=150)
    ap.add_argument("--restore-share", type=float, default=0.15)
    ap.add_argument("--drop-share", type=float, default=0.5)
    a = ap.parse_args()
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / a.model / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "legal"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("addr").alias("a_addr"), "ctry",
        pl.col("legal").alias("a_legal"))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "addr", "ctry", "is_domain", "has_alias", "legal"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("addr").alias("b_addr"),
        pl.col("ctry").alias("ctry_b"), "is_domain", "has_alias", pl.col("legal").alias("b_legal")) for s in (2, 3)])
    pp = pl.read_parquet(P["work"] / "output" / a.model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    raw = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
    m = pl.read_csv(P["work"] / "output" / a.run_in / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb")).join(s1.select("e1", "q"), on="e1")
             .join(pool.select("eb", "pid"), on="eb").select("q", "pid"))
    tabs = []
    fr_d = None
    for c in ("us", "india", "france"):
        s1c, poolc = s1.filter(pl.col("ctry") == c), pool.filter(pl.col("ctry_b") == c)
        rare = rare_vocab(s1c["a_addr"], poolc["b_addr"])
        s1c = street(s1c, "q", "a_addr", "a_st", rare).with_columns(house(pl.col("a_addr")).alias("a_h"), pl.len().over("a_core").alias("n_ns"))
        rc = raw.join(s1c.select("q"), on="q", how="semi")
        pc = street(poolc.join(rc.select("pid"), on="pid", how="semi"), "pid", "b_addr", "b_st", rare).with_columns(house(pl.col("b_addr")).alias("b_h"))
        d = rc.join(s1c, on="q").join(pc, on="pid")
        d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
        d = d.with_columns(pl.when(pl.col("st") == 1).then(pl.lit("mis")).when(pl.col("st") == 2).then(pl.lit("unk"))
                           .when(pl.col("a_h").is_not_null() & (pl.col("a_h") == pl.col("b_h"))).then(pl.lit("same"))
                           .when(pl.col("a_h").is_not_null() & pl.col("b_h").is_not_null()).then(pl.lit("hdiff")).otherwise(pl.lit("unk")).alias("addr"))
        vocab = pl.Series(s1c["a_core"].str.split(" ").explode().unique().drop_nulls())
        d = kinds(d, vocab).with_columns(band(pl.col("p")).alias("pb"))
        n_s1 = s1c.height
        tabs.append(d.group_by("kind", "addr", "pb").agg((pl.len() * 1000.0 / n_s1).alias(f"r_{c}"), pl.len().alias(f"n_{c}")))
        if c == "france":
            la = pl.col("a_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
            lb = pl.col("b_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
            d = d.with_columns(((la.list.len() > 0) & (lb.list.len() > 0) & (la.list.set_intersection(lb).list.len() == 0)).fill_null(False).alias("legal_conflict"),
                               (pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src"))
            fr_d = d.join(fin.with_columns(pl.lit(True).alias("kept")), on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False))
        print(f"{c}: {d.height} raw pairs, {n_s1} S1", flush=True)
    t = tabs[0].join(tabs[1], on=["kind", "addr", "pb"], how="full", coalesce=True).join(tabs[2], on=["kind", "addr", "pb"], how="full", coalesce=True).fill_null(0)
    t = t.with_columns(pl.max_horizontal("r_us", "r_india").alias("ref_hi"), pl.min_horizontal("r_us", "r_india").alias("ref_lo"))
    t = t.with_columns((pl.max_horizontal(pl.col("r_france") - pl.col("ref_hi"), pl.lit(0.0)) / pl.col("r_france")).fill_nan(0.0).alias("share_lo"),
                       (pl.max_horizontal(pl.col("r_france") - pl.col("ref_lo"), pl.lit(0.0)) / pl.col("r_france")).fill_nan(0.0).alias("share_hi"))
    kd = fr_d.group_by("kind", "addr", "pb").agg(pl.col("kept").sum().alias("fr_kept"), (~pl.col("kept")).sum().alias("fr_dropped"))
    t = t.join(kd, on=["kind", "addr", "pb"], how="left").fill_null(0).sort("n_france", descending=True)
    with pl.Config(tbl_rows=120, tbl_width_chars=250, float_precision=3):
        print(t.filter(pl.col("n_france") >= 100).select("kind", "addr", "pb", "n_france", "fr_kept", "fr_dropped", "r_france", "r_us", "r_india", "share_lo", "share_hi"), flush=True)
    n_fr = fr_d.height
    ok_restore = t.filter((pl.col("n_france") >= a.min_n) & (pl.col("share_hi") <= a.restore_share) & ~pl.col("kind").is_in(["typechange"]))
    ok_drop = t.filter((pl.col("n_france") >= a.min_n) & (pl.col("share_lo") >= a.drop_share))
    cand_r = fr_d.filter(~pl.col("kept") & ~pl.col("legal_conflict") & ~((pl.col("kind") == "exact") & (pl.col("addr") == "mis") & (pl.col("n_ns") >= 6))).join(
        ok_restore.select("kind", "addr", "pb", "share_hi"), on=["kind", "addr", "pb"])
    cand_d = fr_d.filter(pl.col("kept")).join(ok_drop.select("kind", "addr", "pb", "share_lo"), on=["kind", "addr", "pb"])
    # slot caps for restores
    kept_n = fr_d.filter(pl.col("kept")).join(cand_d.select("q", "pid"), on=["q", "pid"], how="anti").group_by("q", "src").len().rename({"len": "n"})
    cand_r = cand_r.join(kept_n, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("p", descending=True)
    cand_r = cand_r.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
    per = 100.0 / n_fr

    def gain(x: pl.DataFrame, s: str, restore: bool) -> float:
        if x.height == 0:
            return 0.0
        sh = x[s].to_numpy()
        if restore:  # pairs added: true share 1 - sh
            return float(((1 - sh) * 0.0022 - sh * 0.0063).sum() * per)
        return float((sh * 0.0063 - (1 - sh) * 0.0022).sum() * per)

    gr, gd = gain(cand_r, "share_hi", True), gain(cand_d, "share_lo", False)
    print(f"\nrestore: {cand_r.height} pairs in {ok_restore.height} cells; expected France F0.5 {gr:+.4f} (overall {0.15 * gr:+.5f})")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(cand_r.group_by("kind", "addr", "pb").agg(pl.len(), pl.col("share_hi").first()).sort("len", descending=True))
    print(f"drop: {cand_d.height} pairs in {ok_drop.height} cells; expected France F0.5 {gd:+.4f} (overall {0.15 * gd:+.5f})")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(cand_d.group_by("kind", "addr", "pb").agg(pl.len(), pl.col("share_lo").first()).sort("len", descending=True))
    names = pl.read_parquet(pq / "source1.parquet", columns=["rid", "business_name", "business_address"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "business_name", "business_address")
    pn = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "business_name", "business_address"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                    pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
    for nm, x in (("RESTORE", cand_r), ("DROP", cand_d)):
        smp = x.sample(n=min(25, x.height), seed=4).join(names, on="q").join(pn, on="pid")
        print(f"\n{nm} samples")
        for r in smp.iter_rows(named=True):
            print(f"  p {r['p']:.4f} {r['kind']}/{r['addr']} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    for tag, drop_on in (("R", False), ("RD", True)):
        out = fin
        if drop_on:
            out = out.join(cand_d.select("q", "pid"), on=["q", "pid"], how="anti")
        out = pl.concat([out, cand_r.select("q", "pid").join(out, on=["q", "pid"], how="anti")])
        lists = out.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
        res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
        dst = P["work"] / "output" / f"{a.run_out}_{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "matching_results.tsv", "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for e1, eb in res.select("e1", "eb").iter_rows():
                f.write(f"{e1}\t{eb}\n")
        shutil.copy(P["work"] / "output" / a.run_in / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
        print(f"wrote {dst}: pairs {fin.height} -> {out.height}", flush=True)


if __name__ == "__main__":
    main()
