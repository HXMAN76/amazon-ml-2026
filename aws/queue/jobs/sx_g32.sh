set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_x /home/ec2-user/SageMaker/ber_x --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_x
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber_x/src
conda activate ber
# s32 = s29 + mmBERT cross-encoder score (trained and scored on Sai's account, shared bucket) as the 4th cross-encoder feature xs4
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work/xencG_v7
for i in $(seq 1 240); do aws s3 ls $H/train_xs.parquet >/dev/null 2>&1 && aws s3 ls $H/test_xs.parquet >/dev/null 2>&1 && break; sleep 60; done
mkdir -p $BER_WORK/xencG_v7
aws s3 cp $H/test_xs.parquet $BER_WORK/xencG_v7/test_xs.parquet --only-show-errors
aws s3 cp $H/train_xs.parquet $BER_WORK/xencG_v7/train_xs.parquet --only-show-errors
python -c "import polars as pl; [print(s, pl.read_parquet('$BER_WORK/xencG_v7/'+s+'_xs.parquet').describe()) for s in ('train','test')]"
A="--base v7 --tag _g --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2SymE_v7 --xenc-dir2 xenc3Q_v7 --xenc-dir3 xenc2F_v7 --xenc-dir4 xencG_v7"
python -m ber.stages.stack build --split train $A
python -m ber.stages.stack tfidf --split train --tag _g
python -m ber.stages.stack build --split test $A
python -m ber.stages.stack tfidf --split test --tag _g
python -m ber.stages.stack train --name s32 --base v7 --tag _g --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
python -m ber.stages.stack predict --name s32
python src/scripts/check_submission.py $BER_WORK/output/s32 $BER_DATA/test
python src/scripts/paired_models.py s29 s32
OUT=s3://sagemaker-us-east-1-567503593043/runs
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $BER_WORK/output/s32/$f $OUT/s32/output/$f --only-show-errors; done
aws s3 cp $BER_WORK/models/s32/holdout.json $OUT/s32/holdout.json --only-show-errors
# France recipe (barani's rules, code_v8) then our post-rules (code_x)
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_v8 /home/ec2-user/SageMaker/ber_v8 --delete --exclude 'work/*' --only-show-errors --exact-timestamps
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05
(cd /home/ec2-user/SageMaker/ber_v8 && PYTHONPATH=/home/ec2-user/SageMaker/ber_v8/src python src/scripts/france_variants.py s32 v8w_s32_ARtL --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,legalx:1.01,protect:0.9,$R" --cap 2>&1 | grep -E "^rule|total|wrote|Error|Traceback")
python src/scripts/france_post.py s32 v8w_s32_ARtL v8w_s32_ARtLNC --rules nsaway:6,coined
for n in v8w_s32_ARtL v8w_s32_ARtLNC; do python src/scripts/check_submission.py $BER_WORK/output/$n $BER_DATA/test | tail -2; for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $BER_WORK/output/$n/$f $OUT/$n/output/$f --only-show-errors; done; done
