# v6 step f2: cross-encoder v3 (all 900k S1, 2 epochs), stack bs_w4 paired against bs_w3, set model, blend bs_final4, test weighting, errors
# Stages are Makefile targets (the Makefile at the repository root); checkpoints to the common bucket after each stage.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
[ -d $SM/work/xenc_prev ] || cp -r $SM/work/xenc $SM/work/xenc_prev  # keep the existing v2 backup (for the comparison and a rollback)
export STK=bs_w4 STK_TAG=w4 SETM=bs_set4 FINAL=bs_final4 PREV=bs_w3
STAGES="${STAGES:-bs_xenc bs_final_stack bs_set bs_tw bs_errors}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-f2_xenc3-$RUN" \
python $SM/tools/entry.py
