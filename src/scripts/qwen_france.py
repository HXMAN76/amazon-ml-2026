"""Does Qwen separate France's decoys better than the stack probability? Usage: python src/scripts/qwen_france.py MODEL
For the predicted pairs of France and US (control), the slot-limit fit (see decoy_by_category.py) of the decoy share by relation (exact name, one word swapped, other), probability zone
(below / above 0.995, the France threshold of the recipe) and Qwen score bin (not scored, < 0.05, 0.05 to 0.5, 0.5 to 0.9, >= 0.9). A bin that is decoy-rich in France and clean in the US
is a rule candidate: drop those pairs (worth it above a decoy share of about 26%)."""

import json
import sys

import polars as pl

from ber import config, decision
from pool_support_scan import slot_fit
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s29"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    qs = pl.read_parquet(P["work"] / "xenc3Q_v7" / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("qx"))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left").join(qs, on=["q", "pid"], how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    ex = d.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    rels = {"exact": pl.col("core_eq"), "swap": pl.col("swap") & ~pl.col("core_eq"), "other": ~pl.col("swap") & ~pl.col("core_eq")}
    zones = {"p<0.995": pl.col("p") < 0.995, "p>=0.995": pl.col("p") >= 0.995}
    q = pl.col("qx")
    bins = {"qwen n/a": q.is_null(), "qwen<0.05": q < 0.05, "0.05-0.5": (q >= 0.05) & (q < 0.5), "0.5-0.9": (q >= 0.5) & (q < 0.9), "qwen>=0.9": q >= 0.9}
    print(f"model {name}: predicted pairs {d.height}, with a Qwen score {int(d['qx'].is_not_null().sum())}")
    for c in ("france", "us"):
        base = s1.filter(pl.col("ctry") == c).join(pl.DataFrame({"src": [2, 3]}), how="cross").join(ex, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        dc = d.filter(pl.col("ctry") == c)
        rows = []
        for rn, rc in rels.items():
            for zn, zc in zones.items():
                for bn, bc in bins.items():
                    cnt = dc.filter(rc & zc & bc).group_by("q", "src").len().rename({"len": "n"})
                    per = base.join(cnt, on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
                    n = int(per["n"].sum())
                    if n < 300:
                        continue
                    rows.append([rn, zn, bn, n] + [slot_fit(per.filter(pl.col("src") == s), cap)[2] for s, cap in ((2, 5), (3, 6))])
        with pl.Config(tbl_rows=60, tbl_width_chars=160):
            print(f"\n== {c}: fitted decoy share (slot limit) by relation x zone x Qwen bin\n{pl.DataFrame(rows, schema=['relation', 'zone', 'qwen bin', 'pairs', 'share_S2', 'share_S3'], orient='row')}")


if __name__ == "__main__":
    main()
