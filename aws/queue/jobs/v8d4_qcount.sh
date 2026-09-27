# v8 (CPU lane jobs2): how many pairs the Qwen scoring has to do, and how far it is (elapsed time of the scoring process).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python -c "
import polars as pl
for s in ('test', 'train'):
    print(s, pl.scan_parquet('$BER_WORK/xenc3Q_v7/%s.parquet' % s).select(pl.len()).collect().item())
"
ps -eo pid,etime,cmd | grep "xenc score" | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ls -la $BER_WORK/xenc3Q_v7/
