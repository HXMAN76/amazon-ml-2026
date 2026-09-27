# v8 (CPU lane jobs2): what the protected cut-off still drops in France, by kind, with the holdout true share of the same kinds in the same band.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/band_kinds.py s27 0.995
