"""S1 with exactly one predicted pair: what is that pair? Usage: python src/scripts/single_pair.py VARIANT MODEL
France has about 1% more S1 with one match and 0.4% fewer empty S1 than the US and India (profile_counts.py): some of those lone pairs may be decoys on
singleton S1. For France and the US: the lone pair's kind (exact core, noise-word copy, one-word swap, other), its probability zone and address evidence;
for the holdout the same with the true share of the lone pair (where labels exist)."""

import json
import sys

import numpy as np
import polars as pl

from ber import config, decision
from ber.split import holdout_q

PID_BASE = 10_000_000
NOISE = ["fils", "groupe", "services", "developpement", "france"]


def lone(own: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    one = own.group_by("q").len().filter(pl.col("len") == 1).select("q")
    d = own.join(one, on="q", how="semi").join(s1, on="q", how="left").join(pool, on="pid", how="left")
    a, b = pl.col("a_core"), pl.col("b_core")
    ta, tb = a.str.split(" ").list.unique(), b.str.split(" ").list.unique()
    miss, extra = ta.list.set_difference(tb), tb.list.set_difference(ta)
    kind = (pl.when(a == b).then(pl.lit("exact"))
            .when((miss.list.len() <= 1) & (extra.list.len() >= 1) & extra.list.eval(pl.element().is_in(NOISE)).list.all()).then(pl.lit("noise_copy"))
            .when((miss.list.len() == 1) & (extra.list.len() == 1)).then(pl.lit("one_swap"))
            .otherwise(pl.lit("other")))
    return d.with_columns(kind.alias("kind"), pl.col("p").cut([0.9, 0.99, 0.999], left_closed=True).cast(pl.String).alias("pz"))


def main() -> None:
    var, model = sys.argv[1], sys.argv[2]
    P = config.paths()
    thr = json.loads((P["work"] / "models" / model / "config.json").read_text())["threshold"]
    hthr = json.loads((P["work"] / "models" / model / "holdout.json").read_text())["stack_threshold"]

    def names(split):
        pq = P["parquet"] / split
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), pl.col("core1").alias("a_core"), "ctry")
        pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).select((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"),
                                                                                                    pl.col("core1").alias("b_core")) for s in (2, 3)])
        return s1, pool

    s1, pool = names("test")
    pp = pl.read_parquet(P["work"] / "output" / var / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr)
    d = lone(own, s1, pool)
    s1h, poolh = names("train")
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    ph = pl.read_parquet(P["work"] / "models" / model / "holdout_pred.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64)).join(hq, on="q", how="semi")
    h = lone(decision.assign_exclusive(ph).filter(pl.col("p") >= hthr), s1h, poolh)
    with pl.Config(tbl_rows=40, tbl_cols=8, tbl_width_chars=200):
        for c in ("france", "us"):
            x = d.filter(pl.col("ctry") == c)
            n = s1.filter(pl.col("ctry") == c).height
            print(f"\n{c}: {x.height} S1 with one predicted pair ({x.height / n:.4f} of S1)")
            print(x.group_by("kind", "pz").agg(pl.len().alias("S1"), (pl.len() / n).alias("share_of_all_S1")).sort("kind", "pz"))
        print(f"\nholdout: {h.height} S1 with one predicted pair; true share of the lone pair")
        print(h.group_by("kind", "pz").agg(pl.len().alias("S1"), pl.col("label").mean().alias("true_share")).sort("kind", "pz"))


if __name__ == "__main__":
    main()
