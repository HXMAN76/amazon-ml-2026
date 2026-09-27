set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate pytorch
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
R=s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
W=$BER_WORK
mkdir -p $W/output/s29 $W/models/s29 $W/output/v8w_s29_FIN $W/parquet/test $BER_DATA/test $W/official
for s in 1 2 3; do aws s3 cp $H/parquet/test/source$s.parquet $W/parquet/test/source$s.parquet --only-show-errors & done
for s in 1 2 3; do aws s3 cp $R/dataset/test/test_source$s.tsv $BER_DATA/test/test_source$s.tsv --only-show-errors & done
aws s3 cp $R/utils/validate_submission.py $W/official/validate_submission.py --only-show-errors &
wait
for i in $(seq 1 120); do aws s3 ls $H/s29x/READY >/dev/null 2>&1 && break; sleep 30; done
aws s3 cp $H/s29x/pair_p.parquet $W/output/s29/pair_p.parquet --only-show-errors
aws s3 cp $H/s29x/config.json $W/models/s29/config.json --only-show-errors
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $H/s29x/FIN/$f $W/output/v8w_s29_FIN/$f --only-show-errors; done
for i in $(seq 1 120); do aws s3 ls $H/xfz/france_test_xs.parquet >/dev/null 2>&1 && break; sleep 30; done
aws s3 cp $H/xfz/france_test_xs.parquet $W/xfzin/france_test_xs.parquet --only-show-errors
aws s3 cp $H/xfz/eval_fr_xs.parquet $W/xfzin/eval_fr_xs.parquet --only-show-errors
aws s3 cp $H/xfz/eval_fr.parquet $W/xfzin/eval_fr.parquet --only-show-errors
python src/scripts/xfz_decide.py s29 v8w_s29_FIN v8w_s29_FX $W/xfzin/france_test_xs.parquet $W/xfzin/eval_fr_xs.parquet $W/xfzin/eval_fr.parquet
for t in R RD; do o=$W/output/v8w_s29_FX_$t; python $W/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -2; md5sum $o/matching_results.tsv; for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $o/$f $H/../runs/v8w_s29_FX_$t/output/$f --only-show-errors; done; done
