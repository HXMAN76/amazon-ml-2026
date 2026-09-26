# v8 (GPU lane jobs, before model 4): the team's fitted Qwen3-0.6B cross-encoder (Apache-2.0; never scored for lack of credits) scores its band
# pairs of both splits (xenc3Q_v7), for stack s24 = s22 + xs3 (the team expected +0.0003 to +0.0008 on the holdout: US and India, 85% of the score).
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for d in xenc3Q models/v7 features/train xenc_v7 xenc3Q_v7; do aws s3 sync s3://$B/ber/team_work/$d $BER_WORK/$d --only-show-errors; done
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
python -m ber.stages.xenc score --split test --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192
python -m ber.stages.xenc score --split train --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192
aws s3 cp $BER_WORK/xenc3Q_v7/test_xs.parquet s3://$B/ber/v8/xenc3Q_v7/test_xs.parquet --only-show-errors
aws s3 cp $BER_WORK/xenc3Q_v7/train_xs.parquet s3://$B/ber/v8/xenc3Q_v7/train_xs.parquet --only-show-errors
