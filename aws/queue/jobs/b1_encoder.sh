# v6 step b1_encoder: second name+address encoder (dense_all2) and the first stage bs_w1 at test density
# Stages are Makefile targets (code/business_entity_resolution/Makefile); checkpoints to the common bucket after each stage.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export THIN="--thin"
STAGES="${STAGES:-bs_encoder}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-b1_encoder-$RUN" \
python $SM/tools/entry.py
