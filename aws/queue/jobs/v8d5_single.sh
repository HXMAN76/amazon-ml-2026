# v8 (CPU lane jobs2): S1 with exactly one predicted pair, France against the US and the labelled holdout.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/single_pair.py v8u_s27_AR s27
