# v8 step b (GPU lane jobs): France-aware cross-encoder 1. Synthetic France-style pairs from TRAIN records (initials, spelled legal forms,
# glued words as true copies; one common word swapped as sibling decoys) on top of a replay sample of the team's e5-base fit set;
# warm-started from the team's model (xenc2/model), 1 epoch; scores every France shortlisted test pair and the locked-holdout pairs.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
while [ ! -f $BER_WORK/.v8a_synced ]; do sleep 30; done
python -m pytest -q src/tests/test_xenc_fr.py
python -m ber.stages.xenc_fr --base-dir xenc2_v7 --all-dir xenc2F_v7 --dir xencFR
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python -m ber.stages.xenc train --dir xencFR --base-model $BER_WORK/xenc2/model --model-dir xencFR/model --set epochs=1,lr=0.00001,batch=64
mkdir -p $BER_WORK/xencF1
ln -sf $BER_WORK/xencFR/test.parquet $BER_WORK/xencF1/test.parquet
ln -sf $BER_WORK/xencFR/train.parquet $BER_WORK/xencF1/train.parquet
python -m ber.stages.xenc score --split test --dir xencF1 --model-dir xencFR/model --set score_batch=512
python -m ber.stages.xenc score --split train --dir xencF1 --model-dir xencFR/model --set score_batch=512
aws s3 cp $BER_WORK/xencF1/test_xs.parquet s3://$B/ber/v8/xencF1/test_xs.parquet --only-show-errors
aws s3 cp $BER_WORK/xencF1/train_xs.parquet s3://$B/ber/v8/xencF1/train_xs.parquet --only-show-errors
