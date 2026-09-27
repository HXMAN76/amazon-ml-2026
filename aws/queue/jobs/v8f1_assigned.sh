# v8 (CPU lane jobs2): per country, predicted pairs per S1 and the share of the pool nobody got, for s28 (raw) and v8u_s28_AR (France recipe),
# against train truth (pairs per S1, unowned pool share) and the holdout's own predictions: is France now over- or under-assigning?
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import json, numpy as np, polars as pl
from ber import config, decision
PB = 10_000_000
P = config.paths()
for split in ("train", "test"):
    pq = P["parquet"] / split
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "ctry"]).select(pl.col("rid").cast(pl.Int64).alias("q"), "ctry")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "ctry"]).select((pl.col("rid").cast(pl.Int64) + s * PB).alias("pid"), "ctry") for s in (2, 3)])
    if split == "train":
        lab = pl.read_parquet(pq / "labels.parquet").select(pl.col("s1_rid").cast(pl.Int64).alias("q"), (pl.col("src").cast(pl.Int64) * PB + pl.col("other_rid").cast(pl.Int64)).alias("pid"))
        sets = {"train truth": lab}
    else:
        sets = {}
        for run in ("s28", "v8u_s28_AR"):
            pp = pl.read_parquet(P["work"] / "output" / run / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
            sets[f"test {run}"] = decision.assign_exclusive(pp).filter(pl.col("p") >= 0.72).select("q", "pid")
    for nm, pr in sets.items():
        for c in sorted(s1["ctry"].unique().to_list()):
            n1 = s1.filter(pl.col("ctry") == c).height; npool = pool.filter(pl.col("ctry") == c).height
            k = pr.join(s1.filter(pl.col("ctry") == c), on="q", how="semi")
            print(f"{nm:18s} {c:7s} S1 {n1:8d} pool {npool:8d} pairs {k.height:8d} per S1 {k.height / n1:.4f} pool unowned {1 - k['pid'].n_unique() / npool:.4f} "
                  f"S1 empty {1 - k['q'].n_unique() / n1:.4f}", flush=True)
PY
