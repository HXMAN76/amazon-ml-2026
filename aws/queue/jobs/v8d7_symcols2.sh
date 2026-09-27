# v8 (CPU lane jobs2): check how xenc2SymE_v7 (two symmetric seeds averaged) is built from the seed-0 scores (xenc2Fsym_v7).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
aws s3 cp s3://$B/ber/team_work/xenc2Fsym_v7/test_xs.parquet /tmp/sym1.parquet --only-show-errors
python - <<PY
import numpy as np, polars as pl
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7) / (1 - np.clip(x, 1e-7, 1 - 1e-7)))
e = pl.read_parquet("$BER_WORK/xenc2SymE_v7/test_xs.parquet")
a = pl.read_parquet("/tmp/sym1.parquet").rename({"xs": "xs1", "xs_asym": "asym1"})
print(a.schema)
m = e.join(a, on=["q", "pid"], how="inner").sample(200000, seed=0)
l1, le = lg(m["xs1"].to_numpy()), lg(m["xs"].to_numpy())
l2 = 2 * le - l1
gap = np.abs(l1 - l2)
print("rows", m.height, "corr(seed_gap, |l1-l2|)", np.corrcoef(gap, m["xs_seed_gap"].to_numpy())[0, 1], "median abs diff", np.median(np.abs(gap - m["xs_seed_gap"].to_numpy())))
print("xs_asym vs asym1: corr", np.corrcoef(m["xs_asym"].to_numpy(), m["asym1"].to_numpy())[0, 1])
PY
