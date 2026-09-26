# v8 step a6 (CPU lane jobs2): where cross-encoder 1 disagrees with the old one, slot-fit decoy shares per kind (who is right?).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/xfr_slots.py s22 xencF1
