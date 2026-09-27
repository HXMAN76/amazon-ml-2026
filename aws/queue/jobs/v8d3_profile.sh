# v8 (CPU lane jobs2): match-count profiles per S1: train truth, holdout truth and predictions, test US / India / France predictions (v8u_s27_AR).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/profile_counts.py v8u_s27_AR s27
