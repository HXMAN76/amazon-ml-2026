#!/bin/bash
# Runs as ec2-user: sync code, build the `ber` env, start the job runner. Idempotent.
B=sagemaker-us-east-1-567503593043
mkdir -p /home/ec2-user/SageMaker/ber /home/ec2-user/SageMaker/dataset
aws s3 sync s3://$B/ber/code /home/ec2-user/SageMaker/ber --only-show-errors
aws s3 cp s3://$B/ber/jobrunner.sh /home/ec2-user/jobrunner.sh --only-show-errors
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q '^ber ' || conda create -y -q -n ber python=3.12
conda activate ber
pip install -q -r /home/ec2-user/SageMaker/ber/requirements.txt && python -c "import pandas,scipy,sklearn,lightgbm,rapidfuzz" && echo ready > /home/ec2-user/SageMaker/ber/.env_ready
pgrep -f jobrunner.sh >/dev/null || nohup bash /home/ec2-user/jobrunner.sh > /home/ec2-user/jobrunner.log 2>&1 &
