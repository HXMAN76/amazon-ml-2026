set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_x /home/ec2-user/SageMaker/ber_x --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_x
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber_x/src
conda activate ber
python src/scripts/typo_restore.py s29 v8w_s29_FIN v8w_s29_FINT
o=$BER_WORK/output/v8w_s29_FINT; python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -2
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp $o/$f s3://sagemaker-us-east-1-567503593043/runs/v8w_s29_FINT/output/$f --only-show-errors; done
