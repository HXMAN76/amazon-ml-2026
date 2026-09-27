# v9 (CPU lane jobs2): French-ized copy of the training data (frenchify.py), 12 sample true pairs, then `prepare` on it into work_fr.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/france/frenchify.py $SM/data_fr --sample 12
df -h $SM | tail -1
python src/scripts/france/frenchify.py $SM/data_fr
BER_DATA=$SM/data_fr BER_WORK=$SM/work_fr python -m ber.stages.prepare --split train 2>&1 | tail -6
BER_WORK=$SM/work_fr python - <<'PY'
import polars as pl
from ber import config
pq = config.paths()["parquet"] / "train"
with pl.Config(tbl_rows=10, fmt_str_lengths=45, tbl_width_chars=220):
    print(pl.read_parquet(pq / "source2.parquet", columns=["entity_id", "business_name", "name1", "core1", "legal", "addr", "ctry"]).sample(8, seed=0))
    print("labels", pl.read_parquet(pq / "labels.parquet").height)
PY
