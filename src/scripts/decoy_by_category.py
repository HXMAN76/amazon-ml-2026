"""Decoy share of each kind of France pair, from slot occupancy alone (no labels). Usage: python src/scripts/decoy_by_category.py MODEL
An S1 has at most 5 S2 and 6 S3 matches. Let k be its number of confident exact-name copies in a source. Pairs of a given kind that are true copies fit into
the (cap - k) free slots, so their number per S1 falls linearly to zero at k = cap; decoys arrive at a rate that does not depend on k. Fitting
rate(k) = D + T * (cap - k) / cap per kind (weighted least squares over the S1 counts) gives the decoy rate D, the true rate T at k = 0 and the decoy share
of the kind, D / (D + T * mean_slots_free / cap). A kind whose decoy share is above about 26% is worth dropping (`research.md` section 25)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "a_core", "addr": "a_addr"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = s1.join(s1.group_by("a_addr").len().rename({"len": "addr_n"}), on="a_addr", how="left")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "addr_tset", "house_eq", "legal_conflict")
              .join(d.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    d = flag(d.join(feat, on=["q", "pid"], how="left"), df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    kinds = {
        "swap": pl.col("swap"),
        "legal conflict (no swap)": (~pl.col("swap")) & (pl.col("legal_conflict") > 0.5) & ~pl.col("core_eq"),
        "weak name <80 (no swap)": (~pl.col("swap")) & (pl.col("name_tset") < 80) & ~pl.col("core_eq"),
        "house number differs (no swap)": (~pl.col("swap")) & (pl.col("house_eq") < 0.5) & ~pl.col("core_eq"),
        "tiny pool name": (pl.col("b_core").str.len_chars() <= 3) & ~pl.col("core_eq"),
        "p < 0.99 (no swap)": (~pl.col("swap")) & (pl.col("p") < 0.99) & ~pl.col("core_eq"),
        "any non-exact, non-swap": (~pl.col("swap")) & ~pl.col("core_eq"),
        "exact name but p < 0.999": pl.col("core_eq") & (pl.col("p") < 0.999),
    }
    for c in ("france", "us"):
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        print(f"\n== {c}: fitted decoy share per kind (S2 cap 5, S3 cap 6)")
        rows = []
        for kn, cond in kinds.items():
            cnt = d.filter(pl.col("ctry") == c).filter(cond).group_by("q", "src").len().rename({"len": "n"})
            per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
            out = [kn, int(per["n"].sum())]
            for src, cap in ((2, 5), (3, 6)):
                g = per.filter(pl.col("src") == src).group_by("k").agg(pl.len().alias("w"), pl.col("n").mean().alias("r")).filter(pl.col("k") <= cap).sort("k")
                k = g["k"].to_numpy().astype(float); w = g["w"].to_numpy().astype(float); r = g["r"].to_numpy()
                x = (cap - k) / cap
                A = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
                sol, *_ = np.linalg.lstsq(A, r * np.sqrt(w), rcond=None)
                D, T = max(sol[0], 0.0), max(sol[1], 0.0)
                tot = float((w * r).sum())
                out += [round(D, 4), round(T, 4), round(min(1.0, D * w.sum() / tot), 3) if tot > 0 else float("nan")]
            rows.append(out)
        with pl.Config(tbl_rows=20, tbl_width_chars=200):
            print(pl.DataFrame(rows, schema=["kind", "pairs", "D_S2", "T_S2", "decoy_share_S2", "D_S3", "T_S3", "decoy_share_S3"], orient="row"))


if __name__ == "__main__":
    main()
