"""Revise a France recipe with the French-aware cross-encoder score xsF. Usage: python src/scripts/france/xfz_decide2.py MODEL RUN_IN RUN_OUT XS_FR EVAL_XS EVAL [PRIOR_RESTORE PRIOR_DROP]
XS_FR: (q, pid, xs) of France's test pairs; EVAL_XS / EVAL: the French-ized labelled eval pairs and their scores. The eval set is mostly positive, so\ndecisions use per-bin likelihood ratios with the prior of the set being revised (about 0.5 true among the pairs RUN_IN dropped, 0.95 among those it kept).
Restores raw pairs RUN_IN dropped with xsF above the restore cut, drops kept pairs below the drop cut, never touching the decoy classes that US/India
labels cannot teach (type-word changes, French legal-form conflicts, exact names in another street). Cuts: restored bins reach a posterior >= 0.9, dropped bins <= 0.2; writes RUN_OUT_R (restores) and RUN_OUT_RD (restores + drops)."""

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
    edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]
    n1, n0 = int(e["label"].sum()), int((e["label"] == 0).sum())
    cal = []  # (lo, hi, n_true, n_false, likelihood ratio with add-one smoothing)
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = e.filter((pl.col("xs") >= lo) & (pl.col("xs") < hi))
        t1, t0 = int(x["label"].sum()), int((x["label"] == 0).sum())
        cal.append((lo, hi, t1, t0, ((t1 + 1) / (n1 + 1)) / ((t0 + 1) / (n0 + 1))))
    print(f"calibration on the French-ized eval ({n1} true, {n0} false): lo hi n_true n_false LR post@0.5 post@0.95")
    post = lambda lr, pi: pi * lr / (pi * lr + 1 - pi)  # noqa: E731
    for c in cal:
        print(f"  {c[0]:.3f} {c[1]:.3f} {c[2]:6d} {c[3]:5d} {c[4]:9.3f} {post(c[4], 0.5):.3f} {post(c[4], 0.95):.3f}")
    PI_R, PI_D = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5, float(sys.argv[8]) if len(sys.argv) > 8 else 0.95
    POST_R = float(sys.argv[9]) if len(sys.argv) > 9 else 0.9  # restore bins need this posterior (break-even for F0.5 is about 0.74)
    LOW_PI = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0  # > 0: also add pairs below the model threshold, prior LOW_PI
    SWAP_X = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0  # > 0: drop kept one-word swaps of a type word into an unknown word when xsF < SWAP_X
    cut_r = 1.0
    for lo, hi, t1, t0, lr in reversed(cal):  # extend the restore zone downwards while every bin keeps posterior >= 0.9 at the dropped-set prior
        if post(lr, PI_R) >= POST_R:
            cut_r = lo
        else:
            break
    cut_d = 0.0
    for lo, hi, t1, t0, lr in cal:  # extend the drop zone upwards while every bin keeps posterior <= 0.2 at the kept-set prior
        if post(lr, PI_D) <= 0.2:
            cut_d = hi
        else:
            break
    print(f"restore cut xsF >= {cut_r} (prior {PI_R}), drop cut xsF < {cut_d} (prior {PI_D})", flush=True)

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
    stem = lambda c: c.list.eval(pl.element().str.replace(r"(es|s|e)$", ""))  # noqa: E731  plural / feminine forms: ateliers, culturel(le)
    tw = pl.Series(sorted({t.rstrip("s").removesuffix("e") if not t.endswith("es") else t[:-2] for t in TYPE} | {"culturel", "sporti", "atelier"})).implode()
    d = d.with_columns(stem(pl.col("add")).alias("add"), stem(pl.col("gone")).alias("gone"))
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
    # the French-aware model contradicts measured France evidence on two classes: noise-word copies (groupe / fils / cie / services ...) and
    # copies with an empty pool address; drops never touch them, nor exact names at the S1's street, nor typos (one word, 75%+ similar)
    noise = pl.Series(["groupe", "group", "fil", "ci", "cie", "service", "developpement", "franc", "france", "associ", "et", "and", "holding", "participation",
                       "international", "distribution"]).implode()
    diff_noise = pl.concat_list(pl.col("add"), pl.col("gone")).list.eval(pl.element().is_in(noise)).list.all().fill_null(False)
    empty_b = pl.col("b_addr").fill_null("").str.len_chars() == 0
    same_exact = pl.col("exact") & (pl.col("st") == 0)
    drop = d.filter(pl.col("kept") & (pl.col("xsF") < cut_d) & ~diff_noise & ~empty_b & ~same_exact)
    kept_n = fin_fr.join(drop.select("q", "pid"), on=["q", "pid"], how="anti").with_columns((pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src")).group_by("q", "src").len().rename({"len": "n"})
    rest = rest.join(kept_n, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("xsF", descending=True)
    rest = rest.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
    n_fr = raw.height
    print(f"restores {rest.height} ({100 * rest.height / n_fr:.2f}% of France pairs), drops {drop.height} ({100 * drop.height / n_fr:.2f}%)")
    for nm, x in (("RESTORE", rest), ("DROP", drop)):
        print(f"\n{nm} samples")
        for r in x.sample(n=min(25, x.height), seed=2).iter_rows(named=True):
            print(f"  p {r['p'] if r['p'] is not None else float('nan'):.4f} xsF {r['xsF']:.3f} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    if LOW_PI > 0:
        cut_l = 1.0
        for lo, hi, t1, t0, lr in reversed(cal):
            if post(lr, LOW_PI) >= POST_R:
                cut_l = lo
            else:
                break
        owned = fin.select("pid").unique()
        low = decision.assign_exclusive(pp.join(fr.select("q"), on="q", how="semi")).filter((pl.col("p") < thr) & (pl.col("p") >= 0.02)).select("q", "pid", "p")
        low = low.join(owned, on="pid", how="anti").join(xs, on=["q", "pid"]).filter(pl.col("xsF") >= cut_l).join(fr, on="q").join(frp, on="pid")
        low = street(street(low, "q", "a_addr", "a_st", rare), "pid", "b_addr", "b_st", rare)
        low = low.with_columns(pl.Series("st", compare(low["a_st"].to_list(), low["b_st"].to_list())), house(pl.col("a_addr")).alias("a_h"), house(pl.col("b_addr")).alias("b_h"))
        low = low.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
        low = low.with_columns(stem(pl.col("tb").list.set_difference(pl.col("ta"))).alias("add"), stem(pl.col("ta").list.set_difference(pl.col("tb"))).alias("gone"))
        hdiff = pl.col("a_h").is_not_null() & pl.col("b_h").is_not_null() & (pl.col("a_h") != pl.col("b_h"))
        low = low.with_columns(
            (pl.col("add").list.eval(pl.element().is_in(tw)).list.any() & pl.col("gone").list.eval(pl.element().is_in(tw)).list.any()).fill_null(False).alias("typechange"),
            ((la.list.len() > 0) & (lb.list.len() > 0) & (la.list.set_intersection(lb).list.len() == 0)).fill_null(False).alias("legal_conflict"),
            (pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src"))
        low = low.filter(~pl.col("typechange") & ~pl.col("legal_conflict") & (pl.col("st") != 1) & ~hdiff)
        taken = pl.concat([fin_fr.select("q", "pid"), rest.select("q", "pid")]).with_columns((pl.col("pid") // PID_BASE).cast(pl.Int32).alias("src")).group_by("q", "src").len().rename({"len": "n"})
        low = low.join(taken, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("xsF", descending=True)
        low = low.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
        print(f"\nbelow-threshold additions: cut xsF >= {cut_l} (prior {LOW_PI}): {low.height} pairs; p zones:", low.group_by(pl.col("p").cut([0.1, 0.3, 0.5])).len().sort("p").to_dicts())
        for r in low.sample(n=min(25, low.height), seed=3).iter_rows(named=True):
            print(f"  p {r['p']:.4f} xsF {r['xsF']:.3f} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
        rest = pl.concat([rest.select("q", "pid"), low.select("q", "pid")])
    swapdrop = pl.DataFrame(schema={"q": pl.Int64, "pid": pl.Int64})
    if SWAP_X > 0:
        from rapidfuzz import fuzz
        noise_st = {"groupe", "group", "fil", "ci", "cie", "service", "servic", "cy", "compagni", "partenair", "associat", "developpement", "franc", "france", "associ", "associe", "et", "and", "holding",
                    "participation", "international", "distribution", "sa", "sarl", "sas", "eurl", "sci"}
        cand = d.filter(pl.col("kept") & (pl.col("add").list.len() == 1) & (pl.col("gone").list.len() == 1)).with_columns(
            pl.col("add").list.first().alias("wb"), pl.col("gone").list.first().alias("wa"))
        cand = cand.filter(pl.col("wa").is_in(tw) & ~pl.col("wb").is_in(pl.Series(sorted(noise_st)).implode()) & (pl.col("wb").str.len_chars() >= 3))
        sim = [fuzz.ratio(x, y) for x, y in zip(cand["wa"].to_list(), cand["wb"].to_list())]
        cand = cand.with_columns(pl.Series("wsim", sim, dtype=pl.Float64)).filter(pl.col("wsim") < 70)
        print(f"\ntype word swapped into another word, kept: {cand.height}; by xsF:", cand.group_by(pl.col("xsF").cut([0.1, 0.5, 0.9, 0.99])).len().sort("xsF").to_dicts())
        with pl.Config(tbl_rows=60):
            print("swapped-in words (all kept one-word type swaps):", cand.group_by("wb").len().sort("len", descending=True).head(50).to_dicts())
            print("swapped-out words:", cand.group_by("wa").len().sort("len", descending=True).head(20).to_dicts())
        hi = cand.filter(pl.col("xsF") >= 0.99)
        for r in hi.sample(n=min(30, hi.height), seed=8).iter_rows(named=True):
            print(f"  HI p {r['p']:.4f} xsF {r['xsF']:.3f} {r['wa']}->{r['wb']} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
        # a real word (20+ uses in France's S1 names), not a scrambled typo of the type word: a sibling of another kind
        vc = fr.select(pl.col("a_core").str.split(" ").list.unique().alias("w")).explode("w").drop_nulls().with_columns(
            pl.col("w").str.replace(r"(es|s|e)$", "").alias("w")).group_by("w").len().filter(pl.col("len") >= 20)
        cand = cand.with_columns((pl.col("wb").is_in(vc["w"].implode()) & (pl.col("wb").str.len_chars() >= 5)).alias("real_word"), pl.col("xsF").fill_null(-1.0))
        print("kept type swaps into a real word:", cand.filter(pl.col("real_word")).height, cand.filter(pl.col("real_word")).group_by("wb").len().sort("len", descending=True).head(30).to_dicts())
        rw = cand.filter(pl.col("real_word") & (pl.col("xsF").fill_null(1.0) >= SWAP_X))
        for r in rw.sample(n=min(20, rw.height), seed=9).iter_rows(named=True):
            print(f"  RW p {r['p']:.4f} xsF {r['xsF']:.3f} {r['wa']}->{r['wb']} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
        sd = cand.filter((pl.col("xsF").fill_null(1.0) < SWAP_X) | pl.col("real_word"))
        for r in sd.sample(n=min(25, sd.height), seed=4).iter_rows(named=True):
            print(f"  p {r['p']:.4f} xsF {r['xsF']:.3f} {r['wa']}->{r['wb']} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
        swapdrop = sd.select("q", "pid")
        print(f"swap drops: {swapdrop.height}")
    out_r = pl.concat([fin.join(swapdrop, on=["q", "pid"], how="anti"), rest.select("q", "pid")])
    write(P, s1, pool, out_r, run_in, f"{run_out}_R")
    out_rd = pl.concat([fin.join(drop.select("q", "pid"), on=["q", "pid"], how="anti"), rest.select("q", "pid")])
    write(P, s1, pool, out_rd, run_in, f"{run_out}_RD")


if __name__ == "__main__":
    main()
