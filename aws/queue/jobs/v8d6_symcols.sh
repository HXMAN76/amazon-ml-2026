# v8 (CPU lane jobs2): how the team's averaged symmetric scores (xenc2SymE_v7) relate to the single-seed ones (xenc2Fsym_v7), to write the averaging step
# into reproduce_final.sh.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
H=s3://$B/ber/team_work
aws s3 cp s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work/xenc2Fsym_v7/test_xs.parquet /tmp/sym1.parquet --only-show-errors || echo "no access to the team bucket from the notebook"
python - <<PY
import numpy as np, polars as pl, os
e = pl.read_parquet("$BER_WORK/xenc2SymE_v7/test_xs.parquet")
print(e.schema); print(e.head(5))
if os.path.exists("/tmp/sym1.parquet"):
    a = pl.read_parquet("/tmp/sym1.parquet"); print(a.schema); print(a.head(3))
PY
