set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_x /home/ec2-user/SageMaker/ber_x --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_x
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber_x/src
conda activate ber
OUT=s3://sagemaker-us-east-1-567503593043/runs
post() { n=$1; shift; python src/scripts/france_post.py s29 v8w_s29_ARtL $n --rules "$1"; python src/scripts/check_submission.py $BER_WORK/output/$n $BER_DATA/test | tail -3; aws s3 cp $BER_WORK/output/$n/matching_results.tsv $OUT/$n/output/matching_results.tsv --only-show-errors; aws s3 cp $BER_WORK/output/$n/candidate_pairs.tsv $OUT/$n/output/candidate_pairs.tsv --only-show-errors; }
post v8w_s29_ARtLN "nsaway:6"
post v8w_s29_ARtLNC "nsaway:6,coined"
post v8w_s29_ARtLNaC "nsaway_all:6,coined"
post v8w_s29_ARtLN2C "nsaway:2,coined"
