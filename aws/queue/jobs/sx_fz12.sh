set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate ber
SB=sagemaker-us-east-1-767397931665
H=s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work
aws s3 sync s3://$SB/ber/code /home/ec2-user/SageMaker/ber --delete --exclude 'work/*' --only-show-errors
cd /home/ec2-user/SageMaker/ber
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src
W=$BER_WORK
python src/scripts/glued_typeswap.py s29 v8w_s29_FR_R v8w_s29_FRZ 2>&1 | grep -E "wrote|Traceback|Error|MULTI" 
o=$W/output/v8w_s29_FRZ; python $W/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids | tail -2; md5sum $o/matching_results.tsv
aws s3 cp $o/matching_results.tsv $H/../runs/v8w_s29_FRZ/output/matching_results.tsv --only-show-errors
