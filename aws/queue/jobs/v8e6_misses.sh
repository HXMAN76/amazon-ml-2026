# v8 (CPU lane jobs2): s28's lost true pairs on the holdout (blocking misses, true candidates not kept) by kind, with examples.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/misses.py s28
