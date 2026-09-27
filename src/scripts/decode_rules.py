"""Rule-based decoy removal on top of a model's probabilities (France experiments).

Usage: python src/scripts/decode_rules.py NAME NEWNAME --rule swap|swap_exact [--country france] [--pmax 0.9999]
A predicted pair is a 'swap' when the core names differ by exactly one common word on each side ("nje ecole" against "nje centre"): in France the
sibling businesses at one address look like that. Rule `swap` drops every such pair of the country whose probability is below `pmax`; `swap_exact`
only when the S1 also has a confident exact-name copy (some other predicted pair with the same core name and p >= 0.999). The model's probabilities
are not changed for anything else; the candidate file is unchanged (matches stay a subset). Writes WORK/output/NEWNAME/ like reemit.py."""

import argparse
import json
import time

import polars as pl

from ber import config, decision
from ber.stages.predict import emit
from word_swap import flag, tok_df

PID_BASE = 10_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("newname")
    ap.add_argument("--rule", choices=["swap", "swap_exact"], default="swap")
    ap.add_argument("--country", default="france")
    ap.add_argument("--pmax", type=float, default=0.9999)
    a = ap.parse_args()
    P = config.paths()
    t0 = time.time()
    cfg = json.loads((P["work"] / "models" / a.name / "config.json").read_text())
    thr = cfg["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / a.name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    own = flag(own, df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    exact = own.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q").len().rename({"len": "n_exact"})
    own = own.join(exact, on="q", how="left").with_columns(pl.col("n_exact").fill_null(0))
    drop = (pl.col("ctry") == a.country) & pl.col("swap") & (pl.col("p") < a.pmax)
    if a.rule == "swap_exact":
        drop = drop & (pl.col("n_exact") >= 1)
    dropped = own.filter(drop).select("q", "pid").with_columns(pl.lit(True).alias("_drop"))
    c = own.filter(pl.col("ctry") == a.country).height
    print(f"rule {a.rule}: drops {dropped.height} of {c} predicted {a.country} pairs ({dropped.height / max(c, 1):.4f}); "
          f"{own.filter((pl.col('ctry') == a.country) & pl.col('swap')).height} swaps in total", flush=True)
    out = pp.join(dropped, on=["q", "pid"], how="left").with_columns(pl.when(pl.col("_drop").is_not_null()).then(pl.min_horizontal(pl.col("p"), pl.lit(thr - 1e-6))).otherwise(pl.col("p")).alias("p")).drop("_drop")
    (P["work"] / "output" / a.newname).mkdir(parents=True, exist_ok=True)
    out.write_parquet(P["work"] / "output" / a.newname / "pair_p.parquet", compression="zstd")
    emit(a.newname, P, out, {"threshold": thr, "exclusive": cfg["exclusive"]}, t0)


if __name__ == "__main__":
    main()
