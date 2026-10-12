"""Do copies of one business share their noise? Usage: python src/scripts/sibling_fingerprint.py
If the generator derives S2/S3 copies from a common noisy intermediate, two copies of one S1 agree with each other (raw text,
typos, unit numbers) more often than copies of two namesake S1 do. That would separate the empty-address namesake records
(76% of the missed true pairs, research.md 23.1) whose name alone cannot tell the owner.
Part A: raw-text agreement between two true copies of one S1, against two copies of two S1 with the same core name.
Part B: empty-address pool records whose core name is shared by 2 to 50 S1: how often the owner's other copies are the closest
(raw-name similarity) among all namesake S1's copies; chance is 1 / group size."""

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber import config

PID_BASE = 10_000_000


def main() -> None:
    P = config.paths()
    pq = P["parquet"] / "train"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "business_name", "core1", "ctry"]).select(
        pl.col("rid").cast(pl.Int64).alias("q"), pl.col("business_name").alias("n1"), pl.col("core1").alias("c1"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "business_name", "business_address", "core1", "addr"]).select(
        (pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.col("business_name").alias("nb"), pl.col("business_address").alias("ab"),
        pl.col("core1").alias("cb"), pl.col("addr").alias("an")) for s in (2, 3)])
    lab = pl.read_parquet(pq / "labels.parquet").select(pl.col("s1_rid").cast(pl.Int64).alias("q"),
                                                        (pl.col("other_rid").cast(pl.Int64) + pl.col("src").cast(pl.Int64) * PID_BASE).alias("pid"))
    cp = lab.join(pool, on="pid").join(s1, on="q")
    print(f"true pairs {cp.height}", flush=True)

    # Part A: pairs of copies of one S1 against pairs of copies of two namesake S1 (same core name, different S1)
    a = cp.select("q", "pid", "nb", "ab", "c1", "n1")
    same = a.join(a, on="q", suffix="_j").filter(pl.col("pid") < pl.col("pid_j")).sample(n=2_000_000, seed=1, with_replacement=False)
    ns = s1.group_by("c1").len().filter((pl.col("len") >= 2) & (pl.col("len") <= 50))
    b = a.join(ns.select("c1"), on="c1")
    b = b.with_columns(pl.int_range(pl.len()).shuffle(seed=2).over("c1").alias("_k"))
    b2 = b.with_columns(((pl.col("_k") + 1) % pl.len().over("c1")).alias("_kk"))  # each copy meets the next copy of its group
    other = b.join(b2.select("c1", "_kk", pl.col("q").alias("q_j"), pl.col("pid").alias("pid_j"), pl.col("nb").alias("nb_j"), pl.col("ab").alias("ab_j")),
                   left_on=["c1", "_k"], right_on=["c1", "_kk"]).filter(pl.col("q") != pl.col("q_j"))
    # the shuffle pairs each copy with a random copy of its namesake group; keep only cross-S1 pairs

    def stats(d: pl.DataFrame, tag: str) -> None:
        nb, nbj = d["nb"].to_list(), d["nb_j"].to_list()
        ab, abj = d["ab"].to_list(), d["ab_j"].to_list()
        r_name = process.cpdist(nb, nbj, scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        r_addr = process.cpdist(ab, abj, scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        x = d.with_columns(pl.Series("rn", r_name), pl.Series("ra", r_addr))
        out = x.select(
            pl.len().alias("pairs"),
            (pl.col("nb") == pl.col("nb_j")).mean().alias("raw_name_eq"),
            ((pl.col("nb") == pl.col("nb_j")) & (pl.col("nb") != pl.col("n1"))).mean().alias("name_eq_and_differs_from_s1") if "n1" in x.columns else pl.lit(None).alias("x"),
            (pl.col("nb").str.to_lowercase() == pl.col("nb_j").str.to_lowercase()).mean().alias("lower_name_eq"),
            ((pl.col("ab") == pl.col("ab_j")) & (pl.col("ab") != "")).mean().alias("raw_addr_eq"),
            pl.col("rn").mean().alias("mean_name_ratio"), pl.col("ra").mean().alias("mean_addr_ratio"))
        print(f"\n== {tag}\n{out}", flush=True)

    stats(same, "A1 two copies of ONE S1")
    stats(same.join(ns.select("c1"), on="c1"), "A2 two copies of one S1 whose core name has 2-50 namesake S1")
    stats(other, "A3 copies of TWO namesake S1 (same core name)")

    # Part B: empty-address pool records with a namesake core name
    e = cp.filter((pl.col("an").str.len_chars() == 0) & (pl.col("cb") == pl.col("c1"))).join(ns, on="c1").select("q", "pid", "nb", "c1", pl.col("len").alias("gsize"))
    print(f"\nB: empty-address exact-core namesake pool records: {e.height}", flush=True)
    cand = e.join(s1.select(pl.col("q").alias("y"), "c1"), on="c1")  # every namesake S1 y (owner included)
    cop = cp.filter(pl.col("an").str.len_chars() > 0).select(pl.col("q").alias("y"), pl.col("pid").alias("cpid"), pl.col("nb").alias("cnb"), pl.col("ab").alias("cab"))
    t = cand.join(cop, on="y").filter(pl.col("cpid") != pl.col("pid"))
    t = t.with_columns(pl.Series("r", process.cpdist(t["nb"].to_list(), t["cnb"].to_list(), scorer=fuzz.ratio, dtype=np.float32, workers=-1)),
                       (pl.col("nb") == pl.col("cnb")).alias("eq"))
    g = t.group_by("pid", "q", "y", "gsize").agg(pl.col("r").max().alias("rmax"), pl.col("eq").any().alias("eq_any"))
    g = g.with_columns((pl.col("y") == pl.col("q")).alias("owner"))
    print(g.group_by("owner").agg(pl.len(), pl.col("rmax").mean(), pl.col("eq_any").mean()), flush=True)
    best = g.with_columns(pl.col("rmax").max().over("pid").alias("top"), (pl.col("rmax") == pl.col("rmax").max().over("pid")).sum().over("pid").alias("n_top"))
    own = best.filter(pl.col("owner"))
    res = own.select(pl.len().alias("records_with_owner_copies"),
                     ((pl.col("rmax") == pl.col("top")) & (pl.col("n_top") == 1)).mean().alias("owner_unique_best"),
                     (pl.col("rmax") == pl.col("top")).mean().alias("owner_tied_best"),
                     (1.0 / pl.col("gsize")).mean().alias("chance"),
                     (pl.col("eq_any")).mean().alias("owner_has_raw_equal_copy"))
    print(res, flush=True)
    by = own.with_columns(pl.col("gsize").cut([2, 3, 5, 10, 50]).alias("gs")).group_by("gs").agg(
        pl.len(), ((pl.col("rmax") == pl.col("top")) & (pl.col("n_top") == 1)).mean().alias("unique_best"), (1.0 / pl.col("gsize")).mean().alias("chance")).sort("gs")
    with pl.Config(tbl_rows=20):
        print(by)
        print(t.join(e.select("pid"), on="pid").filter(pl.col("y") == pl.col("q")).select("nb", "cnb", "cab", "r").head(25))


if __name__ == "__main__":
    main()
