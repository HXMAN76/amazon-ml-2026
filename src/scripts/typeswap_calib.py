"""Per-word decoy share of France's swap pairs from the slot limit, and what a lower cut-off than ratio 0.75 would drop. Usage: python src/scripts/typeswap_calib.py MODEL
For every swapped-in word w (the word of the pool name that the S1 name lacks) the pairs per S1 slot at k confident exact copies follow rate(k) = D + T (cap - k) / cap
(decoys do not compete for slots, true copies do); fitted decoy share = D * slots / pairs. A pure-decoy word has A/B (rate at k >= 3 over rate at k = 0) near 1, a
pure-true word near 0.34, so the current typeswap cut-off 0.75 means a decoy share of about 0.62 while the break-even share is 0.26 (ratio about 0.51).
Prints the words by fitted share and the estimated overall gain of dropping every word above a share cut-off (per 1% of France's pairs: +0.0063 France F0.5 if the pairs are
wrong, -0.0022 if true; overall = 0.15 x France)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000
CAPS = {2: 5, 3: 6}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left").filter(pl.col("ctry") == "france")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    base = s1.filter(pl.col("ctry") == "france").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    slots = base.group_by("src", "k").len().rename({"len": "S"})
    sw = d.filter(pl.col("swap")).join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    sw = sw.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    sw = sw.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).list.first().alias("xb"), pl.col("ta").list.set_difference(pl.col("tb")).list.first().alias("xa"))
    cnt = sw.group_by("xb", "src", "k").len().rename({"len": "n"})
    tot_pairs = d.height
    print(f"France predicted pairs {tot_pairs}, swap pairs {sw.height}")
    sA = {s: float(slots.filter((pl.col("src") == s) & (pl.col("k") >= 3))["S"].sum()) for s in (2, 3)}
    sB = {s: float(slots.filter((pl.col("src") == s) & (pl.col("k") == 0))["S"].sum()) for s in (2, 3)}
    rows = []
    for w in sw.group_by("xb").len().filter(pl.col("len") >= 150)["xb"].to_list():
        cw = cnt.filter(pl.col("xb") == w)
        n_tot = int(cw["n"].sum())
        decoys, wsum_all = 0.0, 0.0
        nA = nB = 0.0
        for s, cap in CAPS.items():
            sl = slots.filter((pl.col("src") == s) & (pl.col("k") <= cap)).sort("k")
            k = sl["k"].to_numpy().astype(float)
            S = sl["S"].to_numpy().astype(float)
            nk = np.array([float(cw.filter((pl.col("src") == s) & (pl.col("k") == int(kk)))["n"].sum()) for kk in k])
            r = nk / S
            x = (cap - k) / cap
            a = np.vstack([np.ones_like(x), x]).T * np.sqrt(S)[:, None]
            sol, *_ = np.linalg.lstsq(a, r * np.sqrt(S), rcond=None)
            decoys += max(sol[0], 0.0) * S.sum()
            nA += float(cw.filter((pl.col("src") == s) & (pl.col("k") >= 3))["n"].sum()) / sA[s]
            nB += float(cw.filter((pl.col("src") == s) & (pl.col("k") == 0))["n"].sum()) / sB[s]
        rows.append([w, n_tot, round(min(1.0, decoys / n_tot), 3), round(nA / max(nB, 1e-9), 2)])
    r = pl.DataFrame(rows, schema=["swapped_in_word", "pairs", "fitted_decoy_share", "A_over_B"], orient="row").sort("fitted_decoy_share", descending=True)
    with pl.Config(tbl_rows=80, tbl_width_chars=160):
        print(r.head(70))
    print("\nCut-off table: drop every word with fitted share >= t (words with at least 150 swap pairs)")
    print("t     words  pairs  share_of_France_pairs  mean_share  est_overall_gain (0.15*France F0.5)")
    for t in (0.75, 0.62, 0.5, 0.4, 0.35, 0.3, 0.26):
        x = r.filter(pl.col("fitted_decoy_share") >= t)
        n = int(x["pairs"].sum())
        s = float((x["fitted_decoy_share"] * x["pairs"]).sum() / max(n, 1))
        pct = 100 * n / tot_pairs
        gain = pct * (0.0063 * s - 0.0022 * (1 - s)) * 0.15
        print(f"{t:.2f}  {x.height:5d} {n:6d}  {pct:6.2f}%  {s:.3f}  {gain:+.5f}")
    cur = r.filter(pl.col("A_over_B") >= 0.75)
    print(f"\nwords with A/B >= 0.75 (the current typeswap set, 100+ pairs): {cur.height}, pairs {int(cur['pairs'].sum())}")
    mid = r.filter((pl.col("A_over_B") >= 0.5) & (pl.col("A_over_B") < 0.75))
    print(f"words with 0.5 <= A/B < 0.75: {mid.height}, pairs {int(mid['pairs'].sum())}, mean fitted share {float((mid['fitted_decoy_share'] * mid['pairs']).sum() / max(int(mid['pairs'].sum()), 1)):.3f}")


if __name__ == "__main__":
    main()
