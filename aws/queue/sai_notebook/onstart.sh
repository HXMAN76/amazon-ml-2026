#!/bin/bash
# Notebook lifecycle on-start: job-aware idle auto-stop, then background bootstrap as ec2-user.
B=sagemaker-us-east-1-767397931665
IDLE=3600
curl -sL https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py -o /home/ec2-user/autostop.py
echo "*/5 * * * * [ -f /tmp/job.lock ] || /usr/bin/python3 /home/ec2-user/autostop.py --time $IDLE --ignore-connections" | crontab -u ec2-user -
aws s3 cp s3://$B/ber/bootstrap.sh /home/ec2-user/bootstrap.sh --only-show-errors
chown ec2-user /home/ec2-user/bootstrap.sh /home/ec2-user/autostop.py
nohup sudo -u ec2-user bash /home/ec2-user/bootstrap.sh > /home/ec2-user/onstart.log 2>&1 &
