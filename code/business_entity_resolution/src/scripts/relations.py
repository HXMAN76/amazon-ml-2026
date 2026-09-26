"""Name relation types of France's predicted pairs and their decoy share from slot occupancy. Usage: python src/scripts/relations.py MODEL
Relation of the pool name to the S1 name after removing spaced legal forms (`s a r l`, `e u r l`, `s c i`, `e i`, `s a s`, ...): equal, initials (a pool name of at most 3 letters
that is a subsequence of the initials of the S1 core name), alias (the pool name equals the S1's alias), extra words (S1 tokens are a strict subset of the pool tokens), dropped words
(pool tokens a strict subset of the S1's), swap (one common word replaced), unrelated (no common token), other. For each: pairs, share below p 0.985 and the slot-limit fit of the decoy share
(rate per S1 and source by the number k of confident exact-name copies; see decoy_by_category.py). US as reference."""

import json
import re
import sys

import numpy as np
import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000
SPACED = r"(?:^| )(?:s a r l|s a s u|s a s|e u r l|s c i|s n c|e i r l|e i|s a)(?: |$)"


def strip_spaced(c: pl.Expr) -> pl.Expr:
    """Remove spaced legal forms (two passes: neighbouring matches share a space), collapse blanks."""
    return c.str.replace_all(SPACED, " ").str.replace_all(SPACED, " ").str.replace_all(r"\s+", " ").str.strip_chars()


def subseq(short: str, initials: str) -> bool:
    it = iter(initials)
    return all(c in it for c in short)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s22"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "name2", "ctry"]).rename({"rid": "q", "core1": "a_core", "name2": "a_alias"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = d.with_columns(strip_spaced(pl.col("a_core")).alias("a_n"), strip_spaced(pl.col("b_core")).alias("b_n"))
    d = flag(d.with_columns(pl.col("a_n").alias("_a"), pl.col("b_n").alias("_b")).drop("a_core", "b_core").rename({"a_n": "a_core", "b_n": "b_core"}), df).rename({"a_core": "a_n", "b_core": "b_n"})
    d = d.with_columns(pl.col("a_n").str.split(" ").list.unique().alias("ta"), pl.col("b_n").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns((pl.col("ta").list.set_difference(pl.col("tb")).list.len()).alias("a_only"), (pl.col("tb").list.set_difference(pl.col("ta")).list.len()).alias("b_only"),
                       pl.col("ta").list.set_intersection(pl.col("tb")).list.len().alias("both"))
    ini = d.filter((pl.col("b_n").str.len_chars() <= 3) & ~pl.col("b_n").str.contains(" ") & (pl.col("a_n") != pl.col("b_n"))).select("q", "pid", "a_n", "b_n")
    ok = {(q, pid) for q, pid, a, b in ini.iter_rows() if subseq(b, "".join(t[0] for t in a.split()))}
    d = d.with_columns(pl.struct(["q", "pid"]).map_elements(lambda s: (s["q"], s["pid"]) in ok, return_dtype=pl.Boolean).alias("ini_ok"))
    d = d.with_columns(pl.when(pl.col("a_n") == pl.col("b_n")).then(pl.lit("1 equal (after legal spacing)"))
                         .when(pl.col("ini_ok")).then(pl.lit("2 initials of the S1 name"))
                         .when((pl.col("b_n") == strip_spaced(pl.col("a_alias").fill_null(""))) & (pl.col("a_alias").fill_null("") != "")).then(pl.lit("3 alias of the S1"))
                         .when(pl.col("swap")).then(pl.lit("4 one common word swapped"))
                         .when((pl.col("a_only") == 0) & (pl.col("b_only") > 0)).then(pl.lit("5 extra words in the pool name"))
                         .when((pl.col("b_only") == 0) & (pl.col("a_only") > 0)).then(pl.lit("6 words dropped"))
                         .when(pl.col("both") == 0).then(pl.lit("7 no common token (unrelated or typo)"))
                         .otherwise(pl.lit("8 other")).alias("rel"))
    ex = d.filter((pl.col("rel").str.starts_with("1")) & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    for c in ("france", "us"):
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        rows = []
        for rel in sorted(d["rel"].unique().to_list()):
            x = d.filter((pl.col("ctry") == c) & (pl.col("rel") == rel))
            if x.height < 200:
                continue
            cnt = x.group_by("q", "src").len().rename({"len": "n"})
            per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
            out = [rel, x.height, round(float((x["p"] < 0.985).mean()), 3)]
            for src, cap in ((2, 5), (3, 6)):
                g = per.filter(pl.col("src") == src).group_by("k").agg(pl.len().alias("w"), pl.col("n").mean().alias("r")).filter(pl.col("k") <= cap).sort("k")
                k = g["k"].to_numpy().astype(float); w = g["w"].to_numpy().astype(float); r = g["r"].to_numpy(); xx = (cap - k) / cap
                A = np.vstack([np.ones_like(xx), xx]).T * np.sqrt(w)[:, None]
                sol, *_ = np.linalg.lstsq(A, r * np.sqrt(w), rcond=None)
                Dd = max(sol[0], 0.0); tot = float((w * r).sum())
                out += [[round(float(v), 4) for v in r], round(min(1.0, Dd * w.sum() / tot), 2) if tot > 0 else float("nan")]
            rows.append(out)
        with pl.Config(tbl_rows=20, tbl_width_chars=250, fmt_str_lengths=80):
            print(f"\n== {c}\n", pl.DataFrame(rows, schema=["relation", "pairs", "share_p<0.985", "rate_by_k_S2", "decoy_S2", "rate_by_k_S3", "decoy_S3"], orient="row"))
    with pl.Config(tbl_rows=20, fmt_str_lengths=60, tbl_width_chars=200):
        u = d.filter((pl.col("ctry") == "france") & (pl.col("rel").str.starts_with("7"))).sample(12, seed=1)
        print("\nFrance unrelated-name pairs: a_alias is", "set" if u["a_alias"].is_not_null().any() else "never set")
        print(u.select("p", "a_n", "b_n", "a_alias"))


if __name__ == "__main__":
    main()
