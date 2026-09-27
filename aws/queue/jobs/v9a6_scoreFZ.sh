# v9 (GPU lane, right after v9a4_xfz): score every train/test candidate pair of s28's stack (the xenc2F_v7 pair lists, original text) and the
# French-ized holdout pairs (work_fr hF2F) with the French-aware cross-encoder xencFZ; separate folders so no existing score is overwritten.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export SM
[ -f $SM/work_t/xencFZ/model/config.json ] || { echo "no xencFZ model"; exit 1; }
mkdir -p $SM/work_t/xencFZs $SM/work_fr/hF2FZ
ln -sfn $SM/work_t/xenc2F_v7/train.parquet $SM/work_t/xencFZs/train.parquet
ln -sfn $SM/work_t/xenc2F_v7/test.parquet $SM/work_t/xencFZs/test.parquet
ln -sfn $SM/work_fr/hF2F/train.parquet $SM/work_fr/hF2FZ/train.parquet
export BER_WORK=$SM/work_t
for s in test train; do python -m ber.stages.xenc score --split $s --dir xencFZs --model-dir xencFZ/model --set score_batch=512 2>&1 | grep -E "average precision|scored|Error|Traceback"; done
BER_WORK=$SM/work_fr python -m ber.stages.xenc score --split train --dir hF2FZ --model-dir $SM/work_t/xencFZ/model --set score_batch=512 2>&1 | grep -E "average precision|scored|Error|Traceback"
