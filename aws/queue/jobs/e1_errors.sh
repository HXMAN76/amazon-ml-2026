# v6 step e1_errors: error analysis of the final stack bs_w2 on the locked holdout
# Stages are Makefile targets (code/business_entity_resolution/Makefile); checkpoints to the common bucket after each stage.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
STAGES="${STAGES:-bs_errors}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-e1_errors-$RUN" \
python $SM/tools/entry.py
