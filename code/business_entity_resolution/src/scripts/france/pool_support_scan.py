"""Pool-side name consistency as a France decoy signal. Usage: python src/scripts/france/pool_support_scan.py MODEL
A sibling business (same address, one type word swapped) leaves several pool records with the SAME core name in the S1's candidate list (its own copies in S2 and S3).
The true copies of one S1 carry independent noise, so two of them rarely share an identical name that differs from the S1's. For every predicted pair the script counts
the other candidates of the same S1 with the same core name (sup_same: same source, sup_cross: the other source).
Part 1 (test): fitted decoy share per relation x probability zone x support, France against US (slot-limit fit as in decoy_by_category.py; the US is the control: it has
almost no decoys, so its fitted share is the bias of the fit).
Part 2 (labelled holdout, US and India): true share of the predicted pairs by relation and support: what a support rule would cost."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def with_support(pairs: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """pairs, cand: q, pid, src, b_core, p. cand = every scored candidate; only those with p >= 0.2 count as name-mates."""
    c = cand.filter(pl.col("p") >= 0.2)
    g = c.group_by("q", "b_core", "src").len().rename({"len": "n_src"})
    t = c.group_by("q", "b_core").len().rename({"len": "n_all"})
    d = pairs.join(g, on=["q", "b_core", "src"], how="left").join(t, on=["q", "b_core"], how="left")
    return d.with_columns((pl.col("n_src") - 1).clip(lower_bound=0).fill_null(0).alias("sup_same"), (pl.col("n_all") - pl.col("n_src")).clip(lower_bound=0).fill_null(0).alias("sup_cross")).drop("n_src", "n_all")


def slot_fit(per: pl.DataFrame, cap: int) -> tuple[float, float, float]:
    g = per.group_by("k").agg(pl.len().alias("w"), pl.col("n").mean().alias("r")).filter(pl.col("k") <= cap).sort("k")
    k = g["k"].to_numpy().astype(float)
    w = g["w"].to_numpy().astype(float)
    r = g["r"].to_numpy()
    x = (cap - k) / cap
    a = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(a, r * np.sqrt(w), rcond=None)
    dd, tt = max(sol[0], 0.0), max(sol[1], 0.0)
    tot = float((w * r).sum())
    return round(dd, 4), round(tt, 4), (round(min(1.0, dd * w.sum() / tot), 3) if tot > 0 else float("nan"))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    cand = pp.join(pool, on="pid", how="left")
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = with_support(d, cand)
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    print(f"model {name}, threshold {thr}; predicted pairs {d.height}")
    sup0 = (pl.col("sup_same") == 0) & (pl.col("sup_cross") == 0)
    sup1 = (pl.col("sup_same") >= 1) | (pl.col("sup_cross") >= 1)
    supx = pl.col("sup_cross") >= 1
    supm = (pl.col("sup_same") + pl.col("sup_cross")) >= 2
    rels = {"swap": pl.col("swap"), "other": (~pl.col("swap")) & ~pl.col("core_eq")}
    zones = {"p<0.985": pl.col("p") < 0.985, "p>=0.985": pl.col("p") >= 0.985}
    sups = {"any": pl.lit(True), "sup0": sup0, "sup>=1": sup1, "sup_cross>=1": supx, "sup>=2": supm}
    for c in ("france", "us"):
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        dc = d.filter(pl.col("ctry") == c)
        rows = []
        for rn, rc in rels.items():
            for zn, zc in zones.items():
                for sn, sc in sups.items():
                    cond = rc & zc & sc
                    cnt = dc.filter(cond).group_by("q", "src").len().rename({"len": "n"})
                    per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
                    out = [rn, zn, sn, int(per["n"].sum())]
                    for src, cap in ((2, 5), (3, 6)):
                        out += list(slot_fit(per.filter(pl.col("src") == src), cap)[::2])
                    rows.append(out)
        with pl.Config(tbl_rows=60, tbl_width_chars=200):
            print(f"\n== {c}: fitted decoy share (D/(D+T)-style share of the kind) for S2 (cap 5) and S3 (cap 6)")
            print(pl.DataFrame(rows, schema=["relation", "zone", "support", "pairs", "D_S2", "share_S2", "D_S3", "share_S3"], orient="row"))

    # part 2: labelled holdout (US, India): what does support cost?
    hp = pl.read_parquet(P["work"] / "models" / name / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    pt = P["parquet"] / "train"
    s1t = pl.read_parquet(pt / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    poolt = pl.concat([pl.read_parquet(pt / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    dft = tok_df(s1t.rename({"a_core": "core1"}))
    candt = hp.join(poolt, on="pid", how="left")
    sel = decision.assign_exclusive(hp).filter(pl.col("p") >= thr).join(s1t, on="q", how="left").join(poolt, on="pid", how="left")
    sel = with_support(sel, candt)
    sel = flag(sel, dft).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    rels_h = {"exact": pl.col("core_eq"), **rels}
    rows = []
    for rn, rc in rels_h.items():
        for zn, zc in zones.items():
            for sn, sc in sups.items():
                x = sel.filter(rc & zc & sc)
                rows.append([rn, zn, sn, x.height, round(float(x["label"].mean()), 4) if x.height else float("nan")])
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print("\n== HOLDOUT (labelled, US and India): true share of the predicted pairs")
        print(pl.DataFrame(rows, schema=["relation", "zone", "support", "pairs", "true_share"], orient="row"))


if __name__ == "__main__":
    main()
