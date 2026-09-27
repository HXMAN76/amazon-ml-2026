# v8 (CPU lane jobs2): unclaimed shortlisted pairs below the decision in v8u_s28_AR by kind and address, France against the US and India:
# France predicts 3.31 pairs per S1 against 3.39 (US) with lower precision, so about 40k French true pairs are missing; where do they sit?
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/france_recall.py v8u_s28_AR
