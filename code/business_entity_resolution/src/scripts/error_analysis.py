"""Error analysis of a trained model on its out-of-fold predictions (train sample).

Decomposes the macro F_0.5 loss into: pairs the blocker never proposed (recall ceiling), model errors on the
proposed pairs, and per-segment behaviour; then prints raw-text examples of false positives, false negatives and
blocking misses so the noise patterns can be read by a person.
Usage: BER_WORK=... python scripts/error_analysis.py [model_name]
"""

import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ber import decision  # noqa: E402

W = os.environ.get("BER_WORK", "work")
name = sys.argv[1] if len(sys.argv) > 1 else "v0"
pq = f"{W}/parquet/train/"
pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(240)
BASE = 10_000_000

cfg = json.load(open(f"{W}/models/{name}/config.json"))
thr = cfg["threshold"]
oof = pl.read_parquet(f"{W}/models/{name}/oof.parquet")
lab = pl.read_parquet(pq + "labels.parquet").with_columns(
    (pl.col("src").cast(pl.Int64) * BASE + pl.col("other_rid")).alias("pid"), pl.col("s1_rid").alias("q"))
smp = pl.read_parquet(f"{W}/sample/train_s1.parquet")
qs = smp.select("q" if "q" in smp.columns else pl.col("rid").alias("q"), "ctry", "n_matches")
n_true = qs.select("q", pl.col("n_matches").alias("n_true"))
s1 = pl.read_parquet(pq + "source1.parquet", columns=["rid", "business_name", "business_address", "ctry"]).rename({"rid": "q"})
s2 = pl.read_parquet(pq + "source2.parquet", columns=["rid", "business_name", "business_address", "ctry", "nl_name", "nl_addr", "addr"])
s3 = pl.read_parquet(pq + "source3.parquet", columns=["rid", "business_name", "business_address", "ctry", "nl_name", "nl_addr", "addr"])
pool = pl.concat([s2.with_columns((pl.col("rid").cast(pl.Int64) + 2 * BASE).alias("pid")),
                  s3.with_columns((pl.col("rid").cast(pl.Int64) + 3 * BASE).alias("pid"))]).drop("rid")
owner = lab.select("pid", owner_q="q")  # pid -> owning S1

print(f"model {name}, threshold {thr:.2f}, exclusive={cfg['exclusive']}, {oof.height} scored pairs, {qs.height} S1")


def per_entity(pred: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F0.5 table for a set of predicted pairs."""
    g = pred.group_by("q").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
    d = qs.join(n_true.select("q", "n_true"), on="q", how="left").join(g, on="q", how="left").with_columns(
        pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0))
    prec, rec = pl.col("tp") / pl.col("n_pred").clip(lower_bound=1), pl.col("tp") / pl.col("n_true").clip(lower_bound=1)
    f = ((1.25) * prec * rec / (0.25 * prec + rec)).fill_nan(0.0)
    return d.with_columns(
        pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
          .when((pl.col("n_pred") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise(f).alias("f"))


excl = decision.assign_exclusive(oof) if cfg["exclusive"] else oof
pred = excl.filter(pl.col("p") >= thr)
e_model = per_entity(pred)
e_oracle = per_entity(oof.filter(pl.col("label") == 1))  # perfect matcher restricted to the candidates
F_model, F_oracle = e_model["f"].mean(), e_oracle["f"].mean()
print(f"\n== LOSS DECOMPOSITION\nmacro F0.5 model {F_model:.4f} | oracle on candidates {F_oracle:.4f} "
      f"(loss from blocking recall {1 - F_oracle:.4f}, loss from matcher {F_oracle - F_model:.4f})")
tp_all, n_all = int(oof["label"].sum()), int(qs["n_matches"].sum())
print(f"pair recall of candidates {tp_all / n_all:.4f}; predicted pairs {pred.height}, precision {pred['label'].mean():.4f}, "
      f"recall vs all true {pred['label'].sum() / n_all:.4f}")

d = e_model.join(e_oracle.select("q", pl.col("f").alias("f_oracle")), on="q")
print("\n== BY COUNTRY\n", d.group_by("ctry").agg(pl.len().alias("n"), pl.col("f").mean().alias("f_model"),
                                                  pl.col("f_oracle").mean(), (pl.col("n_true") == 0).mean().alias("singleton_rate")))
d = d.with_columns(pl.when(pl.col("n_true") == 0).then(pl.lit("0 singleton")).when(pl.col("n_true") == 1).then(pl.lit("1"))
                     .when(pl.col("n_true") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")).alias("bucket"))
print("\n== BY NUMBER OF TRUE MATCHES\n", d.group_by("bucket").agg(pl.len().alias("n"), pl.col("f").mean().alias("f_model"),
                                                                    pl.col("f_oracle").mean()).sort("bucket"))
print("\n== SINGLETONS: predicted a match", float(((d["n_true"] == 0) & (d["n_pred"] > 0)).sum() / max((d["n_true"] == 0).sum(), 1)))

# segment by properties of the true matches of each S1
lm = lab.join(pool.select("pid", "nl_name", "nl_addr", "addr"), on="pid", how="left").group_by("q").agg(
    (pl.col("nl_name") > 0.5).any().alias("any_nonlatin_name"), (pl.col("addr") == "").any().alias("any_empty_addr"))
d2 = d.join(lm, on="q", how="left").with_columns(pl.col("any_nonlatin_name").fill_null(False), pl.col("any_empty_addr").fill_null(False))
print("\n== BY MATCH PROPERTIES\n", d2.group_by("any_nonlatin_name", "any_empty_addr").agg(
    pl.len().alias("n"), pl.col("f").mean().alias("f_model"), pl.col("f_oracle").mean()).sort("any_nonlatin_name", "any_empty_addr"))

print("\n== F0.5 vs THRESHOLD (exclusive)")
rows = []
for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.63, 0.7, 0.8, 0.9):
    pr = excl.filter(pl.col("p") >= t)
    rows.append((t, per_entity(pr)["f"].mean(), pr["label"].mean(), pr["label"].sum() / n_all, pr.height))
print(pl.DataFrame(rows, schema=["thr", "macro_f05", "precision", "recall", "n_pred"], orient="row"))

# ---- error taxonomy
wrong = pred.filter(pl.col("label") == 0).join(owner, on="pid", how="left")
print(f"\n== FALSE POSITIVES {wrong.height}: pool record has another owner S1: {wrong['owner_q'].is_not_null().mean():.3f} "
      f"| distractor (no owner): {wrong['owner_q'].is_null().mean():.3f}")
fn_c = oof.filter(pl.col("label") == 1).join(pred.select("q", "pid").with_columns(pl.lit(1).alias("kept")), on=["q", "pid"], how="left")
fn_c = fn_c.filter(pl.col("kept").is_null())
print(f"== FALSE NEGATIVES among candidates {fn_c.height} ({fn_c.height / max(tp_all, 1):.3f} of found true pairs); "
      f"p quantiles {fn_c['p'].quantile(0.1):.3f} {fn_c['p'].quantile(0.5):.3f} {fn_c['p'].quantile(0.9):.3f}")
lost_excl = fn_c.join(excl.select("pid", pl.col("q").alias("q_win"), pl.col("p").alias("p_win")), on="pid", how="left").filter(
    pl.col("q_win").is_not_null() & (pl.col("q_win") != pl.col("q")))
print(f"   of which lost to exclusive assignment (another S1 scored higher for the same record): {lost_excl.height}")
missed = lab.join(oof.select("q", "pid").with_columns(pl.lit(1).alias("cand")), on=["q", "pid"], how="left").filter(
    pl.col("cand").is_null()).join(smp.select(pl.col("rid").alias("q") if "rid" in smp.columns else "q"), on="q", how="semi")
print(f"== BLOCKING MISSES {missed.height}")


def show(df: pl.DataFrame, title: str, n: int = 12, cols=("p",)) -> None:
    """Print raw-text examples for a sample of pairs."""
    j = (df.sample(min(n, df.height), seed=3).join(s1, on="q", how="left").join(pool, on="pid", how="left", suffix="_r"))
    print(f"\n--- {title}")
    for r in j.iter_rows(named=True):
        ptxt = " ".join(f"{c}={r[c]:.2f}" for c in cols if c in r and r[c] is not None)
        print(f"S1: {r['business_name']!r} | {r['business_address']!r} | {r['ctry']}\n    R : {r['business_name_r']!r} | {r['business_address_r']!r}  {ptxt}")


show(wrong.filter(pl.col("owner_q").is_not_null()), "FALSE POSITIVES whose record belongs to a different S1")
show(wrong.filter(pl.col("owner_q").is_null()), "FALSE POSITIVES that are distractors (no owner)")
show(fn_c, "FALSE NEGATIVES: true pair in candidates but p below threshold or lost")
show(missed.with_columns(pl.lit(None, dtype=pl.Float64).alias("p")), "BLOCKING MISSES: true pair never proposed")
