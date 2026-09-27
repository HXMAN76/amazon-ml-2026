set -ex
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
aws s3 sync s3://sagemaker-us-east-1-567503593043/ber/code_x /home/ec2-user/SageMaker/ber_x --delete --exclude 'work/*' --only-show-errors --exact-timestamps
cd /home/ec2-user/SageMaker/ber_x
export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber_x/src
conda activate ber
python src/scripts/france_explore.py s29
