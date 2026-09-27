# v8 (CPU lane jobs2): France pairs of s28 in [0.995, 0.9999) that thrpn would spare: kinds by source with slot-fit decoy share against the holdout's
# true share, and S3 examples at full against free slots (is S3's elevated decoy share a real decoy kind?).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/band_kinds.py s28 0.9999 0.995
