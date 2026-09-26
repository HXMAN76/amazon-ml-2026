# v6 step a1 (resume after the dense folder fix): the rest of the s6 baseline rebuild.
# WORK stays on the notebook volume (no full mirror); each stage's new files go to the common bucket as a checkpoint.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
STAGES="${STAGES:-bs_dense bs_dense_all bs_first bs_stack}"
RUN=bs
set +x
BER_CKPT_BASE=2 BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-a1b-$RUN" \
python $SM/tools/entry.py
