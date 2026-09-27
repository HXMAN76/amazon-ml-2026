set -ex
# Option 3 on Sai's account: a new cross-encoder family (jhu-clsp/mmBERT-base, MIT, ModernBERT, 307M) fine-tuned on the same
# 966k fit pairs as the e5-base and Qwen cross-encoders (xenc2/train_fit.parquet), 1 epoch, then the band pairs of s29's Qwen feature
# (xenc3Q_v7/{test,train}.parquet) are scored. Fallback if the model cannot load: Qwen/Qwen3-Reranker-0.6B on 400k fit pairs.
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate pytorch
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
aws s3 sync s3://$SB/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
mkdir -p $BER_WORK/xencG $BER_WORK/xencG_v7 $BER_DATA
[ -f $BER_WORK/xencG/train_fit.parquet ] || aws s3 cp $H/xenc2/train_fit.parquet $BER_WORK/xencG/train_fit.parquet --only-show-errors
for s in test train; do [ -f $BER_WORK/xencG_v7/$s.parquet ] || aws s3 cp $H/xenc3Q_v7/$s.parquet $BER_WORK/xencG_v7/$s.parquet --only-show-errors; done
pip install -q transformers sentencepiece polars==1.44.2 duckdb==1.5.5 pyyaml scikit-learn einops xgboost
python -c "import transformers, torch; print(transformers.__version__, torch.__version__, torch.cuda.get_device_name(0))"
M=jhu-clsp/mmBERT-base; X="epochs=1,max_len=128,lr=0.00005"; Y="max_len=128"
python -m ber.stages.xenc train --dir xencG --base-model $M --model-dir xencG/smoke --set $X,max_rows=2000
python -m ber.stages.xenc train --dir xencG --base-model $M --model-dir xencG/model --set $X
echo "$M" > $BER_WORK/xencG/BASE_MODEL
for s in test train; do
  python -m ber.stages.xenc score --split $s --dir xencG_v7 --model-dir xencG/model --set $Y,score_batch=512
  aws s3 cp $BER_WORK/xencG_v7/${s}_xs.parquet s3://$SB/runs/xencG_v7/${s}_xs.parquet --only-show-errors
  aws s3 cp $BER_WORK/xencG_v7/${s}_xs.parquet $H/xencG_v7/${s}_xs.parquet --only-show-errors || echo "shared bucket write failed"
done
aws s3 sync $BER_WORK/xencG/model s3://$SB/runs/xencG/model --only-show-errors
aws s3 cp $BER_WORK/xencG/BASE_MODEL s3://$SB/runs/xencG/BASE_MODEL --only-show-errors
echo READY > /tmp/READY && aws s3 cp /tmp/READY s3://$SB/runs/xencG_v7/READY --only-show-errors
