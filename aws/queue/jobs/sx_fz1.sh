set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate pytorch
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
aws s3 sync s3://$SB/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
W=$BER_WORK; mkdir -p $W/xsrc $W/xenc2/model $W/xfz_eo $W/xfz_ef $W/xfz_fr $BER_DATA
pip install -q transformers sentencepiece polars==1.44.2 duckdb==1.5.5 pyyaml scikit-learn xgboost
aws s3 cp $H/xenc2/train_fit.parquet $W/xsrc/train_fit.parquet --only-show-errors &
aws s3 cp $H/xenc2F_v7/train.parquet $W/xsrc/train_full.parquet --only-show-errors &
aws s3 cp $H/xenc2F_v7/test.parquet $W/xsrc/test_full.parquet --only-show-errors &
aws s3 cp $H/parquet/test/source1.parquet $W/xsrc/source1.parquet --only-show-errors &
aws s3 sync $H/xenc2/model $W/xenc2/model --only-show-errors &
wait
python src/scripts/xfz.py make $W/xfz 90000 210000
python src/scripts/xfz.py evalset $W/xfz 60000
cp $W/xfz/eval_orig.parquet $W/xfz_eo/test.parquet; cp $W/xfz/eval_fr.parquet $W/xfz_ef/test.parquet
# France test pairs (real French text), first-stage p >= 0.05
python -c "
import polars as pl
fr = pl.read_parquet('$W/xsrc/source1.parquet', columns=['rid','ctry']).filter(pl.col('ctry')=='france').select(pl.col('rid').cast(pl.Int64).alias('q'))
t = pl.read_parquet('$W/xsrc/test_full.parquet').with_columns(pl.col('q').cast(pl.Int64)).join(fr, on='q', how='semi').filter(pl.col('p') >= 0.05)
t.write_parquet('$W/xfz_fr/test.parquet'); print('France test pairs', t.height)"
# baseline: the warm-start e5-base model on both eval sets
for e in eo ef; do python -m ber.stages.xenc score --split test --dir xfz_$e --model-dir xenc2/model --set score_batch=512,max_len=128; done
cp $W/xfz_eo/test_xs.parquet $W/xfz/eval_orig_xs.parquet; cp $W/xfz_ef/test_xs.parquet $W/xfz/eval_fr_xs.parquet
echo "== BASELINE (xenc2 e5-base)"; python src/scripts/xfz.py eval $W/xfz
# train the French-aware model (warm start)
python -m ber.stages.xenc train --dir xfz --base-model $W/xenc2/model --model-dir xfz/model --set epochs=1,max_len=128,lr=0.00002
for e in eo ef; do python -m ber.stages.xenc score --split test --dir xfz_$e --model-dir xfz/model --set score_batch=512,max_len=128; done
cp $W/xfz_eo/test_xs.parquet $W/xfz/eval_orig_xs.parquet; cp $W/xfz_ef/test_xs.parquet $W/xfz/eval_fr_xs.parquet
echo "== NEW (xencFZ)"; python src/scripts/xfz.py eval $W/xfz
aws s3 cp $W/xfz/eval_fr_xs.parquet $H/xfz/eval_fr_xs.parquet --only-show-errors; aws s3 cp $W/xfz/eval_fr.parquet $H/xfz/eval_fr.parquet --only-show-errors
python -m ber.stages.xenc score --split test --dir xfz_fr --model-dir xfz/model --set score_batch=512,max_len=128
aws s3 cp $W/xfz_fr/test_xs.parquet $H/xfz/france_test_xs.parquet --only-show-errors
echo DONE
