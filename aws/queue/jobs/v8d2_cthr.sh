# v8 (CPU lane jobs2): per-country thresholds for US and India on s27 (and s22), tuned out-of-fold, checked on the holdout with a paired bootstrap.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/country_thr.py s27
python src/scripts/country_thr.py s22
