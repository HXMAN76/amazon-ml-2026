# v8 step c (GPU lane jobs): France-aware cross-encoder 2 = model 1's synthetic pairs plus 100k "another tenant at the same address"
# negatives (a true copy renamed to another train business of its country, no shared word). Same warm start, epoch and scoring.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python -m ber.stages.xenc_fr --base-dir xenc2_v7 --all-dir xenc2F_v7 --dir xencFR2 --set n_tenant=100000
python -m ber.stages.xenc train --dir xencFR2 --base-model $BER_WORK/xenc2/model --model-dir xencFR2/model --set epochs=1,lr=0.00001,batch=64
mkdir -p $BER_WORK/xencF2
ln -sf $BER_WORK/xencFR2/test.parquet $BER_WORK/xencF2/test.parquet
ln -sf $BER_WORK/xencFR2/train.parquet $BER_WORK/xencF2/train.parquet
python -m ber.stages.xenc score --split test --dir xencF2 --model-dir xencFR2/model --set score_batch=512
python -m ber.stages.xenc score --split train --dir xencF2 --model-dir xencFR2/model --set score_batch=512
aws s3 cp $BER_WORK/xencF2/test_xs.parquet s3://$B/ber/v8/xencF2/test_xs.parquet --only-show-errors
aws s3 cp $BER_WORK/xencF2/train_xs.parquet s3://$B/ber/v8/xencF2/train_xs.parquet --only-show-errors
