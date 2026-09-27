"""Does a France decoy have its own cluster? Usage: python src/scripts/mate_scan.py MODEL [US_SAMPLE_Q]
Hypothesis: a decoy pool record belongs to another business (a sibling at the same address) whose other records are noisy variants of ITS name, so the record is closer to
an outsider at its address (a pool record that is not in the S1's predicted list) than to the S1's own list, while a true noisy copy is closest to the S1's own list.
For every predicted non-exact pair: mate_sim = best name similarity (rapidfuzz ratio of the cores after removing spaced legal forms) to a pool record with the same address
string that is not in the S1's list; list_sim = best similarity to the S1's other listed records. Fitted decoy share (slot-limit fit, see decoy_by_category.py) by
relation x zone x (mate_sim - list_sim) bin, France against US (control)."""

import json
import sys
from collections import defaultdict

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber import config, decision
from pool_support_scan import slot_fit
from word_swap import flag, strip_spaced, tok_df

PID_BASE = 10_000_000
MAX_OUT = 40


def mate_features(d: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """d: predicted pairs (q, pid, b_core, b_addr, ...) of ONE country; pool: pid, b_addr, bc (stripped core)."""
    by_addr = defaultdict(list)
    for pid, addr, bc in pool.select("pid", "b_addr", "bc").iter_rows():
        if addr:
            by_addr[addr].append((pid, bc))
    lists = defaultdict(list)
    for q, pid, bc in d.select("q", "pid", "bc").iter_rows():
        lists[q].append((pid, bc))
    mate, lst = [], []
    for q, pid, addr, bc in d.select("q", "pid", "b_addr", "bc").iter_rows():
        mine = {p for p, _ in lists[q]}
        outs = [c for p, c in by_addr.get(addr, ()) if p != pid and p not in mine][:MAX_OUT] if addr else []
        mate.append(max((fuzz.ratio(bc, c) for c in outs), default=-1.0))
        others = [c for p, c in lists[q] if p != pid]
        lst.append(max((fuzz.ratio(bc, c) for c in others), default=-1.0))
    return d.with_columns(pl.Series("mate_sim", mate, dtype=pl.Float64), pl.Series("list_sim", lst, dtype=pl.Float64))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    us_q = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr"})
    pool = pool.with_columns(strip_spaced(pl.col("b_core")).alias("bc"))
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    rng = np.random.default_rng(0)
    us_ids = s1.filter(pl.col("ctry") == "us")["q"].to_numpy()
    us_keep = set(rng.choice(us_ids, size=min(us_q, len(us_ids)), replace=False).tolist())
    dd = {}
    for c in ("france", "us"):
        dc = d.filter(pl.col("ctry") == c)
        if c == "us":
            dc = dc.filter(pl.col("q").is_in(list(us_keep)))
        dd[c] = mate_features(dc, pool)
        print(f"{c}: {dc.height} predicted pairs scored for mates", flush=True)
    delta = pl.col("mate_sim") - pl.col("list_sim")
    bins = {"no outsider at the address": pl.col("mate_sim") < 0, "outsider, list closer (d<=-10)": (pl.col("mate_sim") >= 0) & (delta <= -10),
            "outsider about equal (-10<d<10)": (pl.col("mate_sim") >= 0) & (delta > -10) & (delta < 10), "outsider closer (d>=10)": (pl.col("mate_sim") >= 0) & (delta >= 10),
            "outsider closer and mate>=85": (pl.col("mate_sim") >= 85) & (delta >= 10)}
    rels = {"swap": pl.col("swap"), "other": (~pl.col("swap")) & ~pl.col("core_eq")}
    zones = {"p<0.985": pl.col("p") < 0.985, "p>=0.985": pl.col("p") >= 0.985}
    for c in ("france", "us"):
        keep = s1.filter(pl.col("ctry") == c)
        if c == "us":
            keep = keep.filter(pl.col("q").is_in(list(us_keep)))
        base = keep.join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        rows = []
        for rn, rc in rels.items():
            for zn, zc in zones.items():
                for bn, bc in bins.items():
                    cnt = dd[c].filter(rc & zc & bc).group_by("q", "src").len().rename({"len": "n"})
                    per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
                    out = [rn, zn, bn, int(per["n"].sum())]
                    for src, cap in ((2, 5), (3, 6)):
                        out += list(slot_fit(per.filter(pl.col("src") == src), cap)[::2])
                    rows.append(out)
        with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=40):
            print(f"\n== {c}: fitted decoy share by (mate_sim - list_sim); D = decoy rate per S1 and source\n{pl.DataFrame(rows, schema=['relation', 'zone', 'outsider bin', 'pairs', 'D_S2', 'share_S2', 'D_S3', 'share_S3'], orient='row')}")


if __name__ == "__main__":
    main()
