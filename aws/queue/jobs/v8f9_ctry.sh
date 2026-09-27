# v8 (CPU lane jobs2): country labels after normalisation, per split and source (is France's country field noisy?).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python - <<'PY'
import polars as pl
from ber import config
P = config.paths()
for split in ("train", "test"):
    for s in (1, 2, 3):
        d = pl.read_parquet(P["parquet"] / split / f"source{s}.parquet", columns=["ctry"])
        print(split, s, d.height, d["ctry"].value_counts(sort=True).head(8).rows())
PY
