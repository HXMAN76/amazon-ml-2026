"""What do the test predictions look like per country, by address-sharing and name-sharing stratum? Unlabelled diagnostic.

Usage: python src/scripts/france_profile.py MODEL
For the predicted pairs (exclusive assignment, model threshold) of the test set: per country and per bucket of S1 that share their address
(1, 2, 3-5, 6+) the share of S1 with a match, matches per S1, mean model probability, the share of pairs whose names agree weakly
(name token-set similarity below 0.5) while the address agrees strongly (house number equal or address token-set similarity above 0.8),
and the mean name similarity. France scores about 0.92 on the portal against about 0.99 for the US and India, so this looks for where
France predictions differ from the US ones.
"""

import json
import sys

import polars as pl

from ber import config, decision


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s17"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "holdout.json").read_text())["stack_threshold"]
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).select("q", "pid", "p")
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "addr_tset", "house_eq", "name_jw", "core_eq")
              .join(own.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    own = own.join(feat, on=["q", "pid"], how="left")
    s1 = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = s1.join(s1.group_by("addr").len().rename({"len": "addr_n"}), on="addr", how="left").join(s1.group_by("core1").len().rename({"len": "core_n"}), on="core1", how="left")
    s1 = s1.with_columns(pl.col("addr_n").cut([1, 2, 5], labels=["a1", "a2", "a3-5", "a6+"]).cast(pl.String).alias("ab"),
                         pl.col("core_n").cut([1, 3, 10], labels=["c1", "c2-3", "c4-10", "c11+"]).cast(pl.String).alias("cb"))
    weak = (pl.col("name_tset") < 0.5) & ((pl.col("house_eq") > 0.5) | (pl.col("addr_tset") > 0.8))
    for key in ("ab", "cb"):
        per_q = own.group_by("q").agg(pl.len().alias("n_pred"), pl.col("p").mean().alias("mean_p"), (pl.col("p") < 0.9).mean().alias("share_p_lt_09"),
                                      weak.mean().alias("share_name_weak_addr_strong"), pl.col("name_tset").mean().alias("mean_name_tset"))
        d = s1.join(per_q, on="q", how="left").with_columns(pl.col("n_pred").fill_null(0))
        g = d.group_by("ctry", key).agg(pl.len().alias("s1"), (pl.col("n_pred") > 0).mean().alias("matched_share"), pl.col("n_pred").mean().alias("matches_per_s1"),
                                        pl.col("mean_p").mean().alias("mean_p"), pl.col("share_p_lt_09").mean().alias("share_p_lt_09"),
                                        pl.col("share_name_weak_addr_strong").mean().alias("weak_name_strong_addr"), pl.col("mean_name_tset").mean().alias("mean_name_tset")).sort("ctry", key)
        with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200):
            print(f"\n== by {key}\n{g}")


if __name__ == "__main__":
    main()
