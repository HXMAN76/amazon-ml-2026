#!/bin/bash
# Notebook lifecycle on-start: idle auto-stop (job-aware), sync code, build `ber` env, start job runner.
B=sagemaker-us-east-1-567503593043
IDLE=3600
curl -sL https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py -o /home/ec2-user/autostop.py
echo "*/5 * * * * [ -f /tmp/job.lock ] || /usr/bin/python3 /home/ec2-user/autostop.py --time $IDLE --ignore-connections" | crontab -u ec2-user -
sudo -u ec2-user -i nohup bash -c "
  mkdir -p /home/ec2-user/SageMaker/ber /home/ec2-user/SageMaker/dataset
  aws s3 sync s3://$B/ber/code /home/ec2-user/SageMaker/ber --only-show-errors
  aws s3 cp s3://$B/ber/jobrunner.sh /home/ec2-user/jobrunner.sh --only-show-errors
  source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
  conda env list | grep -q '^ber ' || conda create -y -q -n ber python=3.11
  conda activate ber
  pip install -q -r /home/ec2-user/SageMaker/ber/requirements.txt
  echo ready > /home/ec2-user/SageMaker/ber/.env_ready
  nohup bash /home/ec2-user/jobrunner.sh > /home/ec2-user/jobrunner.log 2>&1 &
" > /home/ec2-user/onstart.log 2>&1 &
