# v8 step a5 (CPU lane jobs2): what the France S1 look like that the t2c rules leave empty (exact-name best pair at the same address or not,
# namesakes), to decide how far the per-S1 protect option should reach.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/emptied_samples.py s22 s22t2c
