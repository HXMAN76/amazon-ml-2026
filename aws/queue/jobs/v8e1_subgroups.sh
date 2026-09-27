# v8 (CPU lane jobs2): sub-groups where s28 is miscalibrated, found out-of-fold, confirmed on the holdout (drop and restore rules for US/India).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/subgroups.py s28
