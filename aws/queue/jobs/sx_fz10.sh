set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate ber; pip install -q rapidfuzz polars==1.44.2 pyyaml 2>/dev/null || true
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
aws s3 sync s3://$SB/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
W=$BER_WORK
mkdir -p $W/xfzin
for i in $(seq 1 60); do aws s3 ls $H/xfz/france_test_xs.parquet >/dev/null 2>&1 && break; sleep 20; done
for f in france_test_xs eval_fr_xs eval_fr; do aws s3 cp $H/xfz/$f.parquet $W/xfzin/$f.parquet --only-show-errors; done
python -c "import rapidfuzz, polars; print(polars.__version__)"
python src/scripts/xfz_decide2.py s29 v8w_s29_FIN v8w_s29_FR $W/xfzin/france_test_xs.parquet $W/xfzin/eval_fr_xs.parquet $W/xfzin/eval_fr.parquet 0.6 0.95 0.8 0.3 0.5
for t in R; do o=$W/output/v8w_s29_FR_$t; python $W/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -2; md5sum $o/matching_results.tsv; for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $o/$f $H/../runs/v8w_s29_FR_$t/output/$f --only-show-errors; done; done
