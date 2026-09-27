"""France self-training of the stack. Usage: python src/scripts/stack_selftrain.py NAME
Trains XGBoost (s28L's features, settings and number of rounds) on the US/India training rows of WORK/stack_eqL/train plus French test pairs
with PSEUDO-labels (never real labels), chosen by today's per-1,000-S1 evidence against the US/India rates:
  label 1: exact core at the S1's exact address (house_eq > 0.5, addr_tset >= 90) with raw p >= 0.999; noise-word swaps / additions at the exact address with p >= 0.9
           (without them the model treats every common-word swap as a sibling: French noise copies fell to half the US rate);
           typos, glued names and noise-word additions at the exact address with p >= 0.99; coined copies of fb_coined_hi; alias records
           naming the S1.
  label 0: one-word swaps between two common French words (sibling businesses), disjoint legal forms, namesakes on another street
           (fb_ns_ref, fb_nsnear_ref), plain coined names below 0.995 at the exact address; a sample of French candidates with p < 0.02.
French pairs are only training rows here (no folds, no holdout ids). Reports the US/India holdout macro F0.5 against s28L's, then writes output/NAME/pair_p.parquet (France from the
new model, every other country from s28) and models/NAME/config.json (s28's config: threshold 0.72)."""

import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from band_kinds import kinds
from ber import config, decision
from ber.split import holdout_q
from ber.stages.train_gpu import pick_device
from france_variants import NOISE_FR, legal_set

PID_BASE = 10_000_000


def pseudo(P) -> pl.DataFrame:
    W, pq = P["work"], P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "name1", "legal", "ctry"]).filter(pl.col("ctry") == "france").select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), legal_set(pl.col("name1"), pl.col("legal")).alias("la"), "ctry")
    s1 = s1.join(s1.group_by("a_core").len().rename({"len": "n_ns"}), on="a_core")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "name1", "name2", "legal"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("core1").alias("b_core"), pl.col("name1").alias("b_name"),
        pl.col("name2").alias("b_alias"), legal_set(pl.col("name1"), pl.col("legal")).alias("lb")) for s in (2, 3)])
    pp = pl.read_parquet(W / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).join(s1, on="q")                      # France candidates, each record at its best S1
    easy = d.filter(pl.col("p") < 0.02).sample(fraction=1.0, seed=0).head(200_000).select("q", "pid", pl.lit(0).alias("label"), pl.lit("easy_neg").alias("why"))
    k = d.filter(pl.col("p") >= 0.72).join(pool, on="pid")
    fs = sorted(str(f) for f in (W / "features" / "test").glob("part_*.parquet"))
    k = k.join(pl.scan_parquet(fs).select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset").join(k.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect(), on=["q", "pid"], how="left")
    k = kinds(k)
    df = s1.select(pl.col("a_core").str.split(" ").list.unique().alias("t")).explode("t").group_by("t").len().rename({"len": "df"})
    ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
    k = k.with_columns(ta.list.set_difference(tb).list.first().alias("xa"), tb.list.set_difference(ta).list.first().alias("xb"))
    k = k.join(df.rename({"t": "xa", "df": "dfa"}), on="xa", how="left").join(df.rename({"t": "xb", "df": "dfb"}), on="xb", how="left")
    k = k.join(df.select(pl.col("t").alias("b_core"), pl.lit(True).alias("_voc")), on="b_core", how="left")
    lists = {n: pl.read_parquet(W / "v8x" / f"{n}.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).with_columns(pl.lit(True).alias(n))
             for n in ("fb_ns_ref", "fb_nsnear_ref", "fb_coined_hi")}
    for n, x in lists.items():
        k = k.join(x, on=["q", "pid"], how="left").with_columns(pl.col(n).fill_null(False))
    noise = NOISE_FR + ["france", "associes", "and", "et", "cie"]
    ex = (pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90)
    coined = pl.col("b_core").str.contains(r"^[a-z]{6,}$") & pl.col("_voc").is_null()
    alias_ok = (pl.col("b_alias") != "") & (pl.col("a_core").str.split(" ").list.set_difference(pl.col("b_alias").str.split(" ")).list.len() == 0)
    neg = [("sibling_swap", (pl.col("kind") == "one_swap") & (pl.col("dfa") >= 100) & (pl.col("dfb") >= 100) & ~pl.col("xa").is_in(noise) & ~pl.col("xb").is_in(noise)),
           ("legal_disjoint", (pl.col("la").list.len() > 0) & (pl.col("lb").list.len() > 0) & (pl.col("la").list.set_intersection(pl.col("lb")).list.len() == 0)),
           ("namesake", pl.col("fb_ns_ref") | pl.col("fb_nsnear_ref")),
           ("coined_low", coined & ex & (pl.col("p") < 0.995))]
    pos = [("alias", alias_ok), ("coined_hi", pl.col("fb_coined_hi")),
           ("exact_exact", (pl.col("a_core") == pl.col("b_core")) & ex & (pl.col("p") >= 0.999)),
           ("noise_copy", pl.col("kind").is_in(["one_swap", "words_added"]) & pl.col("xb").is_in(noise) & ex & (pl.col("p") >= 0.9)),
           ("copy_forms", pl.col("kind").is_in(["one_typo", "glued_fuzzy"]) & ex & (pl.col("p") >= 0.99)),
           ("noise_added", (pl.col("kind") == "words_added") & pl.col("b_core").str.split(" ").list.set_difference(pl.col("a_core").str.split(" ")).list.eval(pl.element().is_in(noise)).list.all() & ex & (pl.col("p") >= 0.99))]
    out, taken = [], pl.lit(False)
    for why, c in pos:  # positives first: alias records and coined copies outrank the negative cells
        x = k.filter(c & ~taken).select("q", "pid", pl.lit(1).alias("label"), pl.lit(why).alias("why"))
        out.append(x)
        taken = taken | c
    for why, c in neg:
        out.append(k.filter(c & ~taken).select("q", "pid", pl.lit(0).alias("label"), pl.lit(why).alias("why")))
        taken = taken | c
    lab = pl.concat(out + [easy]).unique(["q", "pid"], keep="first")
    print("French pseudo-labels:", sorted(lab.group_by("why", "label").len().rows()), flush=True)
    return lab


def main() -> None:
    name = sys.argv[1]
    P = config.paths()
    W = P["work"]
    ref = W / "models" / "s28L"
    cfg = json.loads((ref / "config.json").read_text())
    feats = cfg["features"]
    rounds = xgb.Booster(model_file=str(ref / "xgb.json")).num_boosted_rounds()
    lab = pseudo(P)
    hold = holdout_q().astype(np.int64)
    xs, ys, hparts = [], [], []
    for f in sorted((W / "stack_eqL" / "train").glob("chunk_*.parquet")):
        d = pl.read_parquet(f).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
        h = pl.Series(np.isin(d["q"].to_numpy(), hold))
        dt = d.filter(~h)
        xs.append(dt.select(feats).to_numpy().astype(np.float32))
        ys.append(dt["label"].to_numpy())
        hparts.append(d.filter(h))
    npseudo = 0
    for f in sorted((W / "stack_eqL" / "test").glob("chunk_*.parquet")):
        d = pl.read_parquet(f).join(lab.select("q", "pid", "label"), on=["q", "pid"], how="inner", suffix="_ps")
        if d.height:
            xs.append(d.select(feats).to_numpy().astype(np.float32)); ys.append(d["label_ps"].to_numpy() if "label_ps" in d.columns else d["label"].to_numpy())
            npseudo += d.height
    x, y = np.concatenate(xs), np.concatenate(ys).astype(np.float32)
    print(f"training rows: {len(y)} ({npseudo} French pseudo-labelled); rounds {rounds}", flush=True)
    p = {"objective": "binary:logistic", "eval_metric": "aucpr", "device": pick_device("auto"), "tree_method": "hist", "max_depth": 9, "eta": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "seed": 0}
    model = xgb.train(p, xgb.QuantileDMatrix(x, label=y, feature_names=feats), rounds)
    del x, y
    # US/India holdout: the new model against s28L at s28L's threshold
    labels = pl.read_parquet(P["parquet"] / "train" / "labels.parquet").group_by("s1_rid").len().rename({"s1_rid": "q", "len": "n_true"}).with_columns(pl.col("q").cast(pl.Int64))
    nt = pl.DataFrame({"q": holdout_q().astype(np.int64)}).join(labels, on="q", how="left").with_columns(pl.col("n_true").fill_null(0))
    hd = pl.concat(hparts)
    hb = hd.select("q", "pid", "label").with_columns(pl.Series("p", model.predict(xgb.DMatrix(hd.select(feats).to_numpy().astype(np.float32), feature_names=feats))))
    ha = pl.read_parquet(ref / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    t = cfg["threshold"]
    fa = decision.per_entity_f05(decision.assign_exclusive(ha).filter(pl.col("p") >= t), nt).sort("q")["f"].to_numpy()
    fb = decision.per_entity_f05(decision.assign_exclusive(hb).filter(pl.col("p") >= t), nt).sort("q")["f"].to_numpy()
    dlt, lo, hi = decision.paired_bootstrap_delta(fa, fb)
    print(f"US/India holdout at {t:.2f}: s28L {fa.mean():.6f}, {name} {fb.mean():.6f}, delta {dlt:+.6f} [{lo:+.6f}, {hi:+.6f}]", flush=True)
    # test: France from the new model, the rest from s28
    fr = set(pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).filter(pl.col("ctry") == "france")["rid"].cast(pl.Int64).to_list())
    parts = []
    for f in sorted((W / "stack_eqL" / "test").glob("chunk_*.parquet")):
        d = pl.read_parquet(f)
        d = d.filter(pl.col("q").is_in(list(fr)))
        parts.append(d.select("q", "pid").with_columns(pl.Series("pn", model.predict(xgb.DMatrix(d.select(feats).to_numpy().astype(np.float32), feature_names=feats)))))
    newfr = pl.concat(parts).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    base = pl.read_parquet(W / "output" / "s28" / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    m = base.join(newfr, on=["q", "pid"], how="left").with_columns(pl.coalesce("pn", "p").cast(pl.Float32).alias("p")).select("q", "pid", "p")
    for sub in ("output", "models"):
        (W / sub / name).mkdir(parents=True, exist_ok=True)
    m.write_parquet(W / "output" / name / "pair_p.parquet", compression="zstd")
    (W / "models" / name / "config.json").write_text((W / "models" / "s28" / "config.json").read_text())
    model.save_model(str(W / "models" / name / "xgb.json"))
    print(f"wrote output/{name}/pair_p.parquet: {m.height} pairs, France pairs rescored {newfr.height}", flush=True)


if __name__ == "__main__":
    main()
