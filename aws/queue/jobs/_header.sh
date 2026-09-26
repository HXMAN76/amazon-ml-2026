# Standard job header (source it first in every job): latest code from S3, env, paths. Same as handoff.md section 4.
set -ex
B=sagemaker-us-east-1-645311222213
SM=/home/ec2-user/SageMaker
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh; conda activate $SM/envs/ber
aws s3 sync s3://$B/ber/code $SM/ber --delete --exclude 'work/*' --only-show-errors
aws s3 sync s3://$B/ber/tools $SM/tools --only-show-errors
cd $SM/ber
export BER_DATA=$SM/dataset BER_WORK=$SM/work PYTHONPATH=$SM/ber/src
