set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate pytorch
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
aws s3 sync s3://$SB/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
W=$BER_WORK
python src/scripts/xfz_decide2.py s29 v8w_s29_FIN v8w_s29_FY $W/xfzin/france_test_xs.parquet $W/xfzin/eval_fr_xs.parquet $W/xfzin/eval_fr.parquet 0.5 0.95
for t in R RD; do o=$W/output/v8w_s29_FY_$t; python $W/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -2; md5sum $o/matching_results.tsv; for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $o/$f $H/../runs/v8w_s29_FY_$t/output/$f --only-show-errors; done; done
