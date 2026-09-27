"""Are France's decoys clean clones of the S1 record? Usage: python src/scripts/clone_probe.py MODEL
If a decoy is the S1 record with one field mutated (and no other noise) its address equals the S1 address exactly, while a true copy carries independent noise.
Compares, for France and US, the share of predicted pairs whose S1 and pool addresses are exactly equal, by relation: exact name, swap with a decoy-type word (fitted decoy share >= 0.6
in typeswap_calib.py), swap with a generic word (share <= 0.25), and other non-exact pairs at p >= 0.985 / < 0.985."""

import json
import sys

import polars as pl

from ber import config, decision
from word_swap import flag, tok_df

PID_BASE = 10_000_000
DECOY_WORDS = ["institut", "parents", "primaire", "college", "maternelle", "elementaire", "danse", "pharmacie", "groupement", "fetes", "section", "culture", "club", "soins", "union", "sportive", "conseil", "societe",
               "comite", "jeunes", "sportif", "compagnie", "loisirs", "anciens", "amicale", "foyer", "ecole", "gestion", "centre"]
GENERIC_WORDS = ["france", "services", "fils", "center", "groupe", "developpement"]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "s27"
    P = config.paths()
    thr = json.loads((P["work"] / "models" / name / "config.json").read_text())["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "legal", "ctry"]).rename({"rid": "q", "core1": "a_core", "addr": "a_addr", "legal": "a_legal"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1", "addr", "legal"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core", "addr": "b_addr", "legal": "b_legal"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    d = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    d = flag(d, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    d = d.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
    d = d.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).list.first().alias("xb"))
    d = d.with_columns((pl.col("a_addr") == pl.col("b_addr")).alias("addr_eq"), (pl.col("a_legal") == pl.col("b_legal")).alias("legal_eq"), (pl.col("b_addr").str.len_chars() == 0).alias("b_addr_empty"),
                       (pl.col("a_addr").str.len_chars() - pl.col("b_addr").str.len_chars()).abs().alias("addr_len_diff"))
    rel = (pl.when(pl.col("core_eq") & (pl.col("p") >= 0.9999)).then(pl.lit("1 exact name, p>=0.9999"))
             .when(pl.col("core_eq")).then(pl.lit("2 exact name, p<0.9999"))
             .when(pl.col("swap") & pl.col("xb").is_in(DECOY_WORDS)).then(pl.lit("3 swap, decoy-type word"))
             .when(pl.col("swap") & pl.col("xb").is_in(GENERIC_WORDS)).then(pl.lit("4 swap, generic word"))
             .when(pl.col("swap")).then(pl.lit("5 swap, other word"))
             .when(pl.col("p") >= 0.985).then(pl.lit("6 other, p>=0.985")).otherwise(pl.lit("7 other, p<0.985")).alias("rel"))
    d = d.with_columns(rel)
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        for c in ("france", "us"):
            x = d.filter(pl.col("ctry") == c).group_by("rel").agg(pl.len().alias("pairs"), pl.col("addr_eq").mean().alias("addr_equal"), pl.col("legal_eq").mean().alias("legal_equal"),
                                                               pl.col("b_addr_empty").mean().alias("pool_addr_empty"), pl.col("addr_len_diff").mean().alias("addr_len_diff"), pl.col("p").mean().alias("mean_p")).sort("rel")
            print(f"\n== {c}\n{x}")


if __name__ == "__main__":
    main()
