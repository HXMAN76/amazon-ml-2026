# v8 step s (CPU lane jobs2): France's confident non-exact, non-swap pairs where decoys concentrate (full S1) against where they are rare (free
# S1): kinds of difference and raw examples, to find the next decoy pattern.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/other_samples.py s22
