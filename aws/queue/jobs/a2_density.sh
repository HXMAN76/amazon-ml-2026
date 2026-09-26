# v6 step a2_density: is the extra test density ownerless copies? baseline stack retrained at test density (portal proxy)
# Stages are Makefile targets (code/business_entity_resolution/Makefile); checkpoints to the common bucket after each stage.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export THIN=""
STAGES="${STAGES:-bs_density bs_thin_stack}"
RUN=bs
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-a2_density-$RUN" \
python $SM/tools/entry.py
