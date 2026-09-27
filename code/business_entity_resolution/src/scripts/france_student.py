"""France student: a classifier trained on France pseudo-labels that decides the pairs the France rules leave to a crude cut.
Usage: python src/scripts/france_student.py MODEL RUN_IN RUN_OUT [--drop T] [--restore T] [--tag NAME]
The organisers allow self-training on the test files. Pseudo-labels come only from signals measured on raw records (EXPERIMENTS.md 12):
  positive  exact core name at the S1's street and house number with p >= 0.99; coined aliases at the S1's exact address; alias records that
            name the S1 ("X Co formerly known as <S1 name>"); noise-word copies (fils, groupe, services, developpement, france) at the exact address
  negative  a type word swapped (learned typeswap vocabulary, fitted decoy share 0.8-0.94); exact name in another street with 6+ namesake S1
            (about 87% decoys); records beyond the 5 S2 / 6 S3 slot cap; candidates the model scores below 0.05 (easy negatives)
Features are record-pair facts that do not define the labels on their own (probability, name/address similarities, edit shapes, rank inside
the S1 and the record, the S1's number of copies at home, address multiplicity, name genericity, source), plus the label signals themselves,
so the student learns how they combine and extends them to the unlabelled pairs: the non-exact pairs that `thrpn` cuts at 0.995 (about 19k of
about 34k are true) and the same-street other-house-number exact pairs.
Decisions (France only, starting from RUN_IN): drop a kept unlabelled pair when the student gives < --drop; restore a raw pair RUN_IN dropped when
the student gives > --restore and it is neither a type swap nor a namesake in another street (slot caps kept). A slot-fit decoy share is printed
for both changed sets (biased, see HANDOFF; a sanity check, not a measurement). 5-fold grouped cross-validation on the pseudo-labels is printed."""

import argparse
import json
import shutil

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber import config, decision
from namesake_street import compare, house, rare_vocab, street
from pool_support_scan import slot_fit
from word_swap import flag, tok_df

PID_BASE = 10_000_000
CAP = {2: 5, 3: 6}
TYPE_WORDS = ["amicale", "amis", "anciens", "atelier", "auto", "cafe", "centre", "club", "college", "comite", "compagnie", "conseil", "culture", "culturelle",
              "danse", "ecole", "ehpad", "elementaire", "federation", "fetes", "foyer", "gestion", "groupement", "institut", "jeunes", "loisirs", "lycee", "maison",
              "maternelle", "medico", "musique", "parents", "patrimoine", "pharmacie", "primaire", "residence", "sante", "section", "service", "societe", "soins",
              "sport", "sportif", "sportive", "union"]
NOISE_WORDS = ["fils", "groupe", "services", "developpement", "france", "associes", "cie"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("run_in")
    ap.add_argument("run_out")
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--restore", type=float, default=0.9)
    a = ap.parse_args()
    P = config.paths()
    pq = P["parquet"] / "test"
    thr = json.loads((P["work"] / "models" / a.model / "config.json").read_text())["threshold"]
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id", "core1", "name1", "legal", "addr", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("entity_id").alias("e1"), pl.col("core1").alias("a_core"), pl.col("name1").alias("a_name"),
        pl.col("legal").alias("a_legal"), pl.col("addr").alias("a_addr"), "ctry")
    fr = s1.filter(pl.col("ctry") == "france")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id", "core1", "legal", "addr", "ctry", "name2", "has_alias"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("entity_id").alias("eb"), pl.col("core1").alias("b_core"), pl.col("legal").alias("b_legal"),
        pl.col("addr").alias("b_addr"), pl.col("ctry").alias("ctry_b"), "name2", "has_alias") for s in (2, 3)])
    frp = pool.filter(pl.col("ctry_b") == "france")
    pp = pl.read_parquet(P["work"] / "output" / a.model / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    pp = pp.join(fr.select("q"), on="q", how="semi")
    # candidates: every France pair, owner-exclusive, with p >= 0.02 (keeps easy negatives out of the bulk)
    cand = decision.assign_exclusive(pp).filter(pl.col("p") >= 0.02).select("q", "pid", "p")
    m = pl.read_csv(P["work"] / "output" / a.run_in / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    fin = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("eb")).join(s1.select("e1", "q"), on="e1")
             .join(pool.select("eb", "pid"), on="eb").select("q", "pid"))
    fin_fr = fin.join(fr.select("q"), on="q", how="semi").with_columns(pl.lit(True).alias("kept"))
    cand = pl.concat([cand, fin_fr.join(cand, on=["q", "pid"], how="anti").join(pp, on=["q", "pid"], how="left").select("q", "pid", "p")])
    print(f"France candidates: {cand.height}, kept by {a.run_in}: {fin_fr.height}", flush=True)

    # features
    rare = rare_vocab(fr["a_addr"], frp["b_addr"])
    s1f = street(fr, "q", "a_addr", "a_st", rare).with_columns(house(pl.col("a_addr")).alias("a_h"), pl.len().over("a_core").alias("n_ns"))
    pc = street(frp.join(cand.select("pid").unique(), on="pid", how="semi"), "pid", "b_addr", "b_st", rare).with_columns(house(pl.col("b_addr")).alias("b_h"))
    d = cand.join(s1f, on="q").join(pc, on="pid").join(fin_fr, on=["q", "pid"], how="left").with_columns(pl.col("kept").fill_null(False))
    d = d.with_columns(pl.Series("st", compare(d["a_st"].to_list(), d["b_st"].to_list())))
    an, bn, aa, ba = d["a_core"].to_list(), d["b_core"].to_list(), d["a_addr"].to_list(), d["b_addr"].to_list()
    d = d.with_columns(pl.Series("name_ratio", process.cpdist(an, bn, scorer=fuzz.ratio, dtype=np.float32, workers=-1)),
                       pl.Series("name_tset", process.cpdist(an, bn, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=-1)),
                       pl.Series("addr_tset", process.cpdist(aa, ba, scorer=fuzz.token_set_ratio, dtype=np.float32, workers=-1)),
                       pl.Series("addr_ratio", process.cpdist(aa, ba, scorer=fuzz.ratio, dtype=np.float32, workers=-1)))
    d = flag(d, tok_df(fr.rename({"a_core": "core1"})))
    d = d.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).alias("add"), pl.col("ta").list.set_difference(pl.col("tb")).alias("gone"))
    tw, nw = pl.Series(TYPE_WORDS), pl.Series(NOISE_WORDS)
    vocab = pl.Series(fr["a_core"].str.split(" ").explode().unique().drop_nulls())
    d = d.with_columns(
        (pl.col("a_core") == pl.col("b_core")).alias("exact"),
        (pl.col("st") == 0).alias("st_same"), (pl.col("st") == 1).alias("st_mis"),
        (pl.col("a_h").is_not_null() & (pl.col("a_h") == pl.col("b_h"))).alias("h_same"),
        (pl.col("a_h").is_not_null() & pl.col("b_h").is_not_null() & (pl.col("a_h") != pl.col("b_h"))).alias("h_diff"),
        pl.col("add").list.eval(pl.element().is_in(tw.implode())).list.any().fill_null(False).alias("add_type"),
        pl.col("gone").list.eval(pl.element().is_in(tw.implode())).list.any().fill_null(False).alias("gone_type"),
        pl.col("add").list.eval(pl.element().is_in(nw.implode())).list.all().fill_null(False).alias("add_noise_only"),
        pl.col("add").list.len().alias("n_add"), pl.col("gone").list.len().alias("n_gone"),
        (pl.col("b_core").str.contains(r"^[a-z]{6,}$") & ~pl.col("b_core").is_in(vocab.implode())).alias("coined"),
        (pl.col("has_alias") & (pl.col("a_core").str.len_chars() > 0) & pl.col("name2").str.contains(pl.col("a_core"), literal=True)).fill_null(False).alias("alias_s1"),
        (pl.col("a_legal") == pl.col("b_legal")).alias("legal_eq"), (pl.col("b_legal") == "").alias("b_no_legal"),
        (pl.col("pid") // PID_BASE).alias("src"),
        pl.col("p").rank("ordinal", descending=True).over("q").alias("rank_q"),
    ).drop("ta", "tb", "add", "gone")
    d = d.with_columns((pl.col("add_type") & pl.col("gone_type")).alias("type_change"))
    home = (pl.col("st_same") & pl.col("h_same") & (pl.col("p") >= 0.99)).cast(pl.Int32)
    d = d.with_columns((home.sum().over("q") - home).alias("home_copies"),
                       (pl.col("exact") & (pl.col("p") >= 0.999)).cast(pl.Int32).sum().over("q", "src").alias("k_exact"),
                       pl.len().over("pid").alias("x"), pl.len().over("b_addr").alias("pool_addr_n"))
    d = d.with_columns(pl.col("src").cast(pl.Int32), pl.col("n_ns").cast(pl.Float32))
    # pseudo-labels
    pos = ((pl.col("exact") & pl.col("st_same") & pl.col("h_same") & (pl.col("p") >= 0.99))
           | (pl.col("coined") & pl.col("st_same") & pl.col("h_same"))
           | pl.col("alias_s1")
           | (pl.col("add_noise_only") & (pl.col("n_gone") == 0) & pl.col("st_same") & pl.col("h_same") & (pl.col("p") >= 0.9)))
    neg = ((pl.col("swap") & pl.col("type_change"))
           | (pl.col("exact") & pl.col("st_mis") & (pl.col("n_ns") >= 6))
           | (pl.col("p") < 0.05))
    d = d.with_columns(pl.when(pos & ~neg).then(1).when(neg & ~pos).then(0).otherwise(None).alias("y"))
    # over-cap: beyond the cap in a source, the lowest-p non-positive pairs are negatives
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("q", "src").alias("_r"))
    over = (pl.col("kept") & (pl.col("_r") > pl.col("src").replace_strict(CAP, return_dtype=pl.Int32)) & pl.col("y").is_null())
    d = d.with_columns(pl.when(over).then(0).otherwise(pl.col("y")).alias("y"))
    feats = ["p", "name_ratio", "name_tset", "addr_tset", "addr_ratio", "exact", "st_same", "st_mis", "h_same", "h_diff", "swap", "add_type", "gone_type",
             "type_change", "add_noise_only", "n_add", "n_gone", "coined", "alias_s1", "legal_eq", "b_no_legal", "src", "rank_q", "home_copies", "k_exact",
             "x", "pool_addr_n", "n_ns"]
    lab = d.filter(pl.col("y").is_not_null())
    print("pseudo-labels:", lab.group_by("y").len().sort("y").to_dicts(), flush=True)
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    X = lab.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()
    y = lab["y"].to_numpy()
    grp = (lab["q"].to_numpy() % 5)
    oof = np.zeros(len(y), dtype=np.float32)
    prm = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", nthread=-1)
    for f in range(5):
        tr, va = grp != f, grp == f
        bst = xgb.train(prm, xgb.DMatrix(X[tr], label=y[tr]), 300)
        oof[va] = bst.predict(xgb.DMatrix(X[va]))
    print(f"grouped 5-fold AUC on pseudo-labels {roc_auc_score(y, oof):.4f}", flush=True)
    bst = xgb.train(prm, xgb.DMatrix(X, label=y), 300)
    imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:12]
    print("top features (gain):", [(feats[int(k[1:])], round(v, 1)) for k, v in imp], flush=True)
    d = d.with_columns(pl.Series("q_st", bst.predict(xgb.DMatrix(d.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()))))
    # decisions
    unl = pl.col("y").is_null()
    raw_pred = pl.col("p") >= thr
    drop = d.filter(pl.col("kept") & unl & (pl.col("q_st") < a.drop)).select("q", "pid")
    rest = d.filter(~pl.col("kept") & raw_pred & (pl.col("q_st") > a.restore) & ~(pl.col("swap") & pl.col("type_change")) & ~(pl.col("exact") & pl.col("st_mis") & (pl.col("n_ns") >= 6)))
    kept_after = d.filter(pl.col("kept")).join(drop, on=["q", "pid"], how="anti").group_by("q", "src").len().rename({"len": "n"})
    rest = rest.join(kept_after, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0)).sort("q_st", descending=True)
    rest = rest.with_columns(pl.int_range(pl.len()).over("q", "src").alias("_i")).filter(pl.col("n") + pl.col("_i") < pl.col("src").replace_strict(CAP, return_dtype=pl.Int32))
    # label-agreement check on the France pairs RUN_IN kept or dropped (how far the student departs from the recipe)
    print(f"student: drops {drop.height} unlabelled kept pairs (q_st < {a.drop}); restores {rest.height} raw pairs RUN_IN dropped (q_st > {a.restore})", flush=True)
    # slot-fit sanity check of the changed sets (per S1 and source, against k exact confident copies)
    base = fr.select("q").join(pl.DataFrame({"src": [2, 3]}, schema={"src": pl.Int32}), how="cross").join(
        d.select("q", "src", "k_exact").unique(["q", "src"]), on=["q", "src"], how="left").with_columns(pl.col("k_exact").fill_null(0).alias("k"))
    for nm, x in (("dropped", drop.join(d.select("q", "pid", "src"), on=["q", "pid"])), ("restored", rest.select("q", "pid", "src"))):
        for s, cap in CAP.items():
            per = base.filter(pl.col("src") == s).join(x.filter(pl.col("src") == s).group_by("q").len().rename({"len": "n"}), on="q", how="left").with_columns(pl.col("n").fill_null(0))
            print(f"  slot fit {nm} S{s}: decoy rate, true rate, decoy share = {slot_fit(per, cap)}", flush=True)
    with pl.Config(tbl_rows=40, fmt_str_lengths=50, tbl_width_chars=220):
        names = pl.read_parquet(pq / "source1.parquet", columns=["rid", "business_name", "business_address"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "business_name", "business_address")
        pn = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "business_name", "business_address"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                        pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab")) for s in (2, 3)])
        for nm, x in (("DROPPED", drop), ("RESTORED", rest.select("q", "pid"))):
            smp = x.sample(n=min(20, x.height), seed=3).join(d.select("q", "pid", "p", "q_st"), on=["q", "pid"]).join(names, on="q").join(pn, on="pid")
            print(f"\n{nm} samples")
            for r in smp.iter_rows(named=True):
                print(f"  p {r['p']:.3f} st {r['q_st']:.3f} | {r['business_name']} | {r['business_address']}\n        -> {r['nb']} | {r['ab']}")
    out = fin.join(drop, on=["q", "pid"], how="anti")
    out = pl.concat([out, rest.select("q", "pid").join(out, on=["q", "pid"], how="anti")])
    print(f"pairs {fin.height} -> {out.height}", flush=True)
    lists = out.join(s1.select("q", "e1"), on="q").join(pool.select("pid", "eb"), on="pid").sort("q", "pid").group_by("e1", maintain_order=True).agg(pl.col("eb").str.join(","))
    res = s1.select("q", "e1").sort("q").join(lists, on="e1", how="left").with_columns(pl.col("eb").fill_null(""))
    dst = P["work"] / "output" / a.run_out
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e1, eb in res.select("e1", "eb").iter_rows():
            f.write(f"{e1}\t{eb}\n")
    shutil.copy(P["work"] / "output" / a.run_in / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    d.select("q", "pid", "p", "q_st", "y", "kept").write_parquet(dst / "student.parquet")
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
