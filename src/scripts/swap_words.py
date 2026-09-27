"""Which words are swapped in the France decoys? Usage: python src/scripts/swap_words.py MODEL
Swap pairs of France (one common core-name word replaced by another, see word_swap.py), grouped by how many confident exact-name copies (k) the S1 already
has in that source: k >= 3 (decoy-rich: only 0 to 2 free slots) against k = 0 (true swaps have all slots to choose from). For the swapped-out word, the swapped-in
word, the position of the swap (first, last, middle token) and the number of core tokens: share in each group and the enrichment ratio. A pattern that
separates the groups separates decoys from true swaps."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left").filter(pl.col("ctry") == "france")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    allq = s1.filter(pl.col("ctry") == "france").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    n_grp = {"A k>=3": allq.filter(pl.col("k") >= 3).height, "B k=0": allq.filter(pl.col("k") == 0).height}
    sw = d.filter(pl.col("swap")).join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    sw = sw.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    sw = sw.with_columns(pl.col("ta").list.set_difference(pl.col("tb")).list.first().alias("xa"), pl.col("tb").list.set_difference(pl.col("ta")).list.first().alias("xb"),
                         pl.col("a_core").str.split(" ").alias("toks"))
    sw = sw.with_columns(pl.col("toks").list.len().alias("ntok"), pl.col("toks").list.eval(pl.element() == pl.element().first()).alias("_x")).drop("_x")
    sw = sw.with_columns(pl.when(pl.col("k") >= 3).then(pl.lit("A k>=3")).when(pl.col("k") == 0).then(pl.lit("B k=0")).otherwise(pl.lit("other")).alias("grp"))
    sw = sw.with_columns(pl.struct(["toks", "xa"]).map_elements(lambda s: ("first" if s["toks"][0] == s["xa"] else "last" if s["toks"][-1] == s["xa"] else "middle") if s["xa"] in s["toks"] else "na", return_dtype=pl.String).alias("pos"))
    print("S1-source slots per group:", n_grp, " swap pairs per group:", sw.group_by("grp").len().to_dicts())
    for col in ("pos", "ntok", "xa", "xb"):
        a = sw.filter(pl.col("grp") == "A k>=3").group_by(col).len().rename({"len": "nA"})
        b = sw.filter(pl.col("grp") == "B k=0").group_by(col).len().rename({"len": "nB"})
        j = a.join(b, on=col, how="full", coalesce=True).with_columns(pl.col("nA").fill_null(0), pl.col("nB").fill_null(0))
        j = j.with_columns((pl.col("nA") / n_grp["A k>=3"]).alias("rate_A"), (pl.col("nB") / n_grp["B k=0"]).alias("rate_B")).with_columns((pl.col("rate_A") / (pl.col("rate_B") + 1e-9)).alias("A_over_B"))
        with pl.Config(tbl_rows=25, tbl_width_chars=160):
            print(f"\n== by {col} (rate per S1-source slot; a decoy-only category has A/B near 1 or above, a true-only one near 0.4 or below)")
            print(j.filter(pl.col("nA") + pl.col("nB") >= 200).sort("A_over_B", descending=True).head(25) if col in ("xa", "xb") else j.sort(col))


if __name__ == "__main__":
    main()
