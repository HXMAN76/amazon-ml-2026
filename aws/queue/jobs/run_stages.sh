# Run pipeline stages (make targets) through entry.py: stage markers, a heartbeat every 2 min, WORK mirrored to S3 after every
# stage, per-stage checkpoints in the common bucket. Edit STAGES / RUN, then `sm.py enqueue aws/queue/jobs/run_stages.sh --name <job>`.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
STAGES="${STAGES:-v5_candidates v5_pass1 v5_expand v5_pass2 v5_xenc v5_stack v5_measure}"
RUN="${RUN:-v5}"
set +x
BER_ML_ROOT=$SM/ml BER_CODE=$SM/ber BER_SKIP_INSTALL=1 BER_STAGES="$STAGES" \
BER_WORK_S3="s3://$B/ber/work/$RUN/" BER_CKPT_S3="s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/$RUN/" \
BER_CKPT_FALLBACK="s3://$B/ber/checkpoints/$RUN/" BER_JOB_NAME="queue-$RUN" \
python $SM/tools/entry.py
