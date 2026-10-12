"""Revise a France recipe with the French-aware cross-encoder score xsF. Usage: python src/scripts/xfz_decide.py MODEL RUN_IN RUN_OUT XS_FR EVAL_XS EVAL
XS_FR: (q, pid, xs) of France's test pairs; EVAL_XS / EVAL: the French-ized labelled eval pairs and their scores (calibration: true share per bin).
Restores raw pairs RUN_IN dropped with xsF above the restore cut, drops kept pairs below the drop cut, never touching the decoy classes that US/India
labels cannot teach (type-word changes, French legal-form conflicts, exact names in another street). Cuts are chosen from the calibration so that a
restored bin is >= 95% true and a dropped bin <= 15% true; writes RUN_OUT_R (restores) and RUN_OUT_RD (restores + drops)."""

import json
import shutil
import sys

import numpy as np
import polars as pl

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}
TYPE = {"amicale", "amis", "anciens", "atelier", "cafe", "centre", "club", "college", "comite", "compagnie", "conseil", "culture", "culturelle", "danse", "ecole",
        "ehpad", "elementaire", "federation", "fetes", "foyer", "gestion", "groupement", "institut", "jeunes", "loisirs", "lycee", "maison", "maternelle", "medico",
        "musique", "parents", "patrimoine", "pharmacie", "primaire", "residence", "sante", "section", "service", "societe", "soins", "sport", "sportif", "sportive",
        "union", "hotel", "theatre", "association", "clinique", "cercle", "collectif", "hopital"}
FR_LEGAL = ["sarl", "sas", "sasu", "eurl", "sci", "snc", "ei", "eirl", "sa", "scop"]


def write(P, s1, pool, pairs, run_in, name):
    lists = pairs.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
    res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
    dst = P["work"] / "output" / name
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e1, eb in res.select("e1", "eb").iter_rows():
            f.write(f"{e1}\t{eb}\n")
    shutil.copy(P["work"] / "output" / run_in / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    print(f"wrote {dst}: {pairs.height} pairs", flush=True)


def main() -> None:
    model, run_in, run_out, xs_fr, ev_xs, ev = sys.argv[1:7]
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    # calibration on the labelled French-ized eval pairs
    e = pl.read_parquet(ev_xs).join(pl.read_parquet(ev, columns=["q", "pid", "label"]), on=["q", "pid"])
    edges = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]
    cal = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = e.filter((pl.col("xs") >= lo) & (pl.col("xs") < hi))
        cal.append((lo, hi, x.height, float(x["label"].mean()) if x.height else float("nan")))
    print("calibration (French-ized eval): lo hi n true_share")
    for c in cal:
        print(f"  {c[0]:.3f} {c[1]:.3f} {c[2]:6d} {c[3]:.3f}")
    ok_restore = [lo for lo, hi, n, t in cal if n >= 50 and t >= 0.95]
    cut_r = min([lo for lo in ok_restore if all(t >= 0.95 for l2, h2, n2, t in cal if l2 >= lo and n2 >= 50)] or [1.0])
    ok_drop = [hi for lo, hi, n, t in cal if n >= 50 and t <= 0.15]
    cut_d = max([hi for hi in ok_drop if all(t <= 0.15 for l2, h2, n2, t in cal if h2 <= hi and n2 >= 50)] or [0.0])
    print(f"restore cut xsF >= {cut_r}, drop cut xsF < {cut_d}", flush=True)

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
    fin_fr = fin.join(fr.select("q"), on="q", how="semi").with_columns(pl.lit(True).alias("kept"))
    xs = pl.read_parquet(xs_fr).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("xsF"))
    d = pl.concat([raw, fin_fr.join(raw, on=["q", "pid"], how="anti").join(pp, on=["q", "pid"], how="left").select("q", "pid", "p")])
    d = d.join(fin_fr, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False)).join(xs, on=["q", "pid"], how="left")
    d = d.join(fr, on="q").join(frp, on="pid")
    rare = rare_vocab(fr["a_addr"], frp["b_addr"])
    d = street(street(d, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
    d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
    d = d.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).alias("add"), pl.col("ta").list.set_difference(pl.col("tb")).alias("gone"))
    tw = pl.Series(sorted(TYPE)).implode()
    la = pl.col("a_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
    lb = pl.col("b_legal").str.split(" ").list.set_intersection(pl.Series(FR_LEGAL).implode())
    d = d.with_columns(
        (pl.col("add").list.eval(pl.element().is_in(tw)).list.any() & pl.col("gone").list.eval(pl.element().is_in(tw)).list.any()).fill_null(False).alias("typechange"),
        ((la.list.len() > 0) & (lb.list.len() > 0) & (la.list.set_intersection(lb).list.len() == 0)).fill_null(False).alias("legal_conflict"),
        ((pl.col("a_core") == pl.col("b_core")) & (pl.col("st") == 1)).alias("exact_away"),
        (pl.col("a_core") == pl.col("b_core")).alias("exact"),
        (pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src"))
    guard = ~pl.col("typechange") & ~pl.col("legal_conflict") & ~pl.col("exact_away")
    xb = pl.col("xsF").cut([0.02, 0.1, 0.3, 0.5, 0.9, 0.98, 0.995]).alias("xs_bin")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(d.filter(pl.col("xsF").is_not_null()).group_by("kept", "exact", xb).len().sort("kept", "exact", "xs_bin"), flush=True)
        print("xsF missing (pair not scored):", d.filter(pl.col("xsF").is_null()).group_by("kept").len().to_dicts())
    rest = d.filter(~pl.col("kept") & guard & (pl.col("xsF") >= cut_r))
    drop = d.filter(pl.col("kept") & (pl.col("xsF") < cut_d))
    kept_n = fin_fr.join(drop.select("q", "pid"), on=["q", "pid"], how="anti").with_columns((pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src")).group_by("q", "src").len().rename({"len": "n"})
    rest = rest.join(kept_n, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("xsF", descending=True)
    rest = rest.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
    n_fr = raw.height
    print(f"restores {rest.height} ({100 * rest.height / n_fr:.2f}% of France pairs), drops {drop.height} ({100 * drop.height / n_fr:.2f}%)")
    for nm, x in (("RESTORE", rest), ("DROP", drop)):
        print(f"\n{nm} samples")
        for r in x.sample(n=min(25, x.height), seed=2).iter_rows(named=True):
            print(f"  p {r['p'] if r['p'] is not None else float('nan'):.4f} xsF {r['xsF']:.3f} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    out_r = pl.concat([fin, rest.select("q", "pid")])
    write(P, s1, pool, out_r, run_in, f"{run_out}_R")
    out_rd = pl.concat([fin.join(drop.select("q", "pid"), on=["q", "pid"], how="anti"), rest.select("q", "pid")])
    write(P, s1, pool, out_rd, run_in, f"{run_out}_RD")


if __name__ == "__main__":
    main()
