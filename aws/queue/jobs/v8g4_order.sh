# v8 (CPU lane jobs2): do file row positions or entity-id numbers of true pairs line up (train labels)? A generator artefact, not a model:
# correlation of S1 row position with the matched record's row position, and of the numeric parts of the entity ids.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import numpy as np, polars as pl
from ber import config
P = config.paths(); pq = P["parquet"] / "train"
lab = pl.read_parquet(pq / "labels.parquet").with_columns(pl.col("s1_rid").cast(pl.Int64), pl.col("other_rid").cast(pl.Int64), pl.col("src").cast(pl.Int64))
s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "entity_id"]).select(pl.col("rid").cast(pl.Int64).alias("s1_rid"), pl.col("entity_id").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("n1"))
n1 = s1.height
for s in (2, 3):
    o = pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "entity_id"]).select(pl.col("rid").cast(pl.Int64).alias("other_rid"), pl.col("entity_id").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("n2"))
    no = o.height
    d = lab.filter(pl.col("src") == s).join(s1, on="s1_rid").join(o, on="other_rid")
    x, y = d["s1_rid"].to_numpy() / n1, d["other_rid"].to_numpy() / no
    rx, ry = d["s1_rid"].rank().to_numpy(), d["other_rid"].rank().to_numpy()
    print(f"source {s}: {d.height} true pairs; row position corr {np.corrcoef(x, y)[0, 1]:+.4f}, rank corr {np.corrcoef(rx, ry)[0, 1]:+.4f}; "
          f"|pos diff| < 0.001: {(np.abs(x - y) < 0.001).mean():.4f} (random ~0.002); id-number corr {np.corrcoef(d['n1'].to_numpy(), d['n2'].to_numpy())[0, 1]:+.4f}", flush=True)
    rnd = np.random.default_rng(0).permutation(len(y))
    print(f"  shuffled baseline |pos diff| < 0.001: {(np.abs(x - y[rnd]) < 0.001).mean():.4f}; id digits: S1 {d['n1'].head(3).to_list()} pool {d['n2'].head(3).to_list()}")
PY
