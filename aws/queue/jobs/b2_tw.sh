# v6 step b2_tw: holdout score and threshold re-weighted to the test mix of countries and look-alike density
# Stages are Makefile targets (code/business_entity_resolution/Makefile); checkpoints to the common bucket after each stage.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
STAGES="${STAGES:-bs_tw}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-b2_tw-$RUN" \
python $SM/tools/entry.py
