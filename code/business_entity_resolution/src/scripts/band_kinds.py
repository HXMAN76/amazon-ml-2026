"""Which kinds of pair does the France cut-off still drop, and how true is each kind where labels exist? Usage: python src/scripts/band_kinds.py MODEL [CUT [LO]]
The protected cut-off (france_variants.py thrpn, default 0.995) drops France pairs with p below CUT that are not equal after spaced legal forms, not
initials and not noise-word copies. Here those pairs are split into kinds (glued with a dropped letter, alias marker "d b a" / "formerly" / "f k a",
same words reordered, no common word = coined alias, one common word swapped, words added, words dropped, other) with counts and examples; the same
kinds among the labelled holdout's predicted pairs in the same probability band give each kind's true share. A kind that is nearly always true on
the holdout and is not decoy-rich in France is a candidate for protection. With LO (a band [LO, CUT) above the cut-off): per kind and source the
France slot-fit decoy share next to the holdout's true share, and S3 examples from S1 with 3 or more confident exact copies in S3 (few free slots:
decoy-rich) against S1 with none."""

import json
import sys
from difflib import SequenceMatcher

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q
from word_swap import strip_spaced

PID_BASE = 10_000_000
NOISE = ["fils", "groupe", "services", "developpement", "associes", "and", "et"]


def names(P, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "name1", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("name1").alias("a_name"), pl.col("core1").alias("a_core"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "name1", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                      pl.col("name1").alias("b_name"), pl.col("core1").alias("b_core")) for s in (2, 3)])
    return s1, pool


def kinds(d: pl.DataFrame) -> pl.DataFrame:
    a, b = pl.col("a_core"), pl.col("b_core")
    ta, tb = a.str.split(" ").list.unique(), b.str.split(" ").list.unique()
    d = d.with_columns(ta.list.set_difference(tb).list.len().alias("_miss"), tb.list.set_difference(ta).list.len().alias("_extra"),
                       ta.list.set_intersection(tb).list.len().alias("_common"))
    ga, gb = d["a_core"].str.replace_all(" ", "").to_list(), d["b_core"].str.replace_all(" ", "").to_list()
    d = d.with_columns(pl.Series("_glue", [SequenceMatcher(None, x, y).ratio() if x and y else 0.0 for x, y in zip(ga, gb)]))
    mw, xw = d.select(ta.list.set_difference(tb).list.first())[:, 0].to_list(), d.select(tb.list.set_difference(ta).list.first())[:, 0].to_list()
    d = d.with_columns(pl.Series("_wsim", [SequenceMatcher(None, x, y).ratio() if x and y else 0.0 for x, y in zip(mw, xw)]))  # one_typo: the swapped words are near-equal
    k = (pl.when(pl.col("b_name").str.contains(r"\b(d b a|dba|formerly|f k a|fka|t a|trading as)\b")).then(pl.lit("alias_marker"))
         .when(~b.str.contains(" ") & (pl.col("_glue") >= 0.85)).then(pl.lit("glued_fuzzy"))
         .when((pl.col("_miss") == 0) & (pl.col("_extra") == 0)).then(pl.lit("reordered"))
         .when(pl.col("_common") == 0).then(pl.lit("no_common_word"))
         .when((pl.col("_miss") == 1) & (pl.col("_extra") == 1) & (pl.col("_wsim") >= 0.7)).then(pl.lit("one_typo"))
         .when((pl.col("_miss") == 1) & (pl.col("_extra") == 1)).then(pl.lit("one_swap"))
         .when((pl.col("_miss") == 0) & (pl.col("_extra") >= 1)).then(pl.lit("words_added"))
         .when((pl.col("_miss") >= 1) & (pl.col("_extra") == 0)).then(pl.lit("words_dropped"))
         .otherwise(pl.lit("other")))
    return d.with_columns(k.alias("kind"))


def protected(d: pl.DataFrame) -> pl.Expr:
    """The pairs thrpn spares: equal after spaced legal forms, initials, noise-word copies (as in france_variants.py)."""
    a, b = pl.col("a_core"), pl.col("b_core")
    ta, tb = a.str.split(" ").list.unique(), b.str.split(" ").list.unique()
    extra = tb.list.set_difference(ta)
    ini = b.str.len_chars().is_between(2, 3) & ~b.str.contains(" ")
    noise = (((ta.list.set_difference(tb).list.len() == 1) & (extra.list.len() == 1) & extra.list.first().is_in(NOISE[:4]))
             | ((ta.list.set_difference(tb).list.len() <= 1) & (extra.list.len() >= 1) & extra.list.eval(pl.element().is_in(NOISE)).list.all()
                & extra.list.eval(pl.element().is_in(NOISE[:5])).list.any()))
    return (strip_spaced(a) == strip_spaced(b)) | ini | noise


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else 0.995
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else thr
    s1, pool = names(P, "test")
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).join(s1, on="q", how="left").filter(pl.col("ctry") == "france").join(pool, on="pid", how="left").with_columns((pl.col("pid") // PID_BASE).alias("src"))
    exs = own.filter((pl.col("a_core") == pl.col("b_core")) & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    slots = s1.filter(pl.col("ctry") == "france").select("q").join(pl.DataFrame({"src": [2, 3]}, schema={"src": pl.Int64}), how="cross").join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    d = own.filter((pl.col("p") >= max(thr, lo)) & (pl.col("p") < cut)).join(slots, on=["q", "src"], how="left")
    d = kinds(d.filter(~protected(d)))
    s1h, poolh = names(P, "train")
    ph = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    hthr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    h = (decision.assign_exclusive(ph.join(pl.DataFrame({"q": holdout_q().astype(np.int64)}), on="q", how="semi")).filter((pl.col("p") >= max(hthr, lo)) & (pl.col("p") < cut))
         .join(s1h, on="q", how="left").join(poolh, on="pid", how="left"))
    h = kinds(h.filter(~protected(h)))
    fr = d.group_by("kind").agg(pl.len().alias("france_pairs"), pl.col("p").mean().alias("france_mean_p"))
    ho = h.group_by("kind").agg(pl.len().alias("holdout_pairs"), pl.col("p").mean().alias("holdout_mean_p"), pl.col("label").mean().alias("holdout_true_share"))
    with pl.Config(tbl_rows=20, tbl_cols=8, tbl_width_chars=200, fmt_str_lengths=40):
        print(f"France pairs of {name} that the protected cut-off {cut} drops (p in [{thr:.2f}, {cut})), by kind; holdout pairs of the same band and kind")
        print(fr.join(ho, on="kind", how="full", coalesce=True).sort("france_pairs", descending=True, nulls_last=True))
        if len(sys.argv) > 3:
            from france_variants import decoy_share

            h = h.with_columns((pl.col("pid") // PID_BASE).alias("src"))
            for kd in d["kind"].unique().sort().to_list():
                x, y = d.filter(pl.col("kind") == kd), h.filter(pl.col("kind") == kd)
                print(f"{kd}: France S2 {x.filter(pl.col('src') == 2).height}, S3 {x.filter(pl.col('src') == 3).height}, slot-fit decoy share S2, S3 {decoy_share(x, slots)}; "
                      f"holdout true share S2 {y.filter(pl.col('src') == 2)['label'].mean()}, S3 {y.filter(pl.col('src') == 3)['label'].mean()}", flush=True)
            for grp, f in (("S3 pairs of S1 with >= 3 confident exact S3 copies", pl.col("k") >= 3), ("S3 pairs of S1 with no exact S3 copy", pl.col("k") == 0)):
                x = d.filter((pl.col("src") == 3) & f)
                print(f"\n{grp}: {x.height}; by kind {sorted(x.group_by('kind').len().rows(), key=lambda r: -r[1])}")
                print(x.sample(min(15, x.height), seed=1).select("p", "kind", "a_name", "b_name"))
            return
        for kd in ("glued_fuzzy", "alias_marker", "reordered", "no_common_word", "words_added", "words_dropped"):
            x = d.filter(pl.col("kind") == kd)
            if x.height:
                print(f"\n{kd}: {x.height} France pairs; examples")
                print(x.sample(min(8, x.height), seed=2).select("p", "a_name", "b_name"))


if __name__ == "__main__":
    main()
