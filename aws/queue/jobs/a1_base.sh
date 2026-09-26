# v6 step a1: rebuild the s6 recipe as the paired baseline (bs_v5 first stage, bs_s6 stack) on this notebook.
# WORK stays on the notebook volume (no full mirror); each stage's new files go to the common bucket as a checkpoint.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
STAGES="${STAGES:-bs_prepare bs_block bs_dense bs_dense_all bs_first bs_stack}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-a1-$RUN" \
python $SM/tools/entry.py
