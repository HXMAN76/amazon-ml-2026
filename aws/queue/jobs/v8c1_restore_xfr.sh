# v8 (CPU lane jobs2): restore candidates by the France-aware cross-encoder score (mean of models 1 and 2), France counts and holdout true shares.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/restore_xfr.py s27 xencF12
python src/scripts/restore_xfr.py s22 xencF12
