#!/bin/bash
# Notebook lifecycle on-start (root, 5-minute limit): job-aware idle auto-stop, then the bootstrap in the background as ec2-user.
# Same design as the team's aws/notebook/onstart.sh; bootstrap runs in a login shell (`-i`), which on this platform is what carries the
# execution-role credentials.
B=sagemaker-us-east-1-645311222213
IDLE=3600
curl -sL https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py -o /home/ec2-user/autostop.py
echo "*/5 * * * * [ -f /tmp/job.lock ] || /usr/bin/python3 /home/ec2-user/autostop.py --time $IDLE --ignore-connections" | crontab -u ec2-user -
aws s3 cp s3://$B/ber/queue/bootstrap.sh /home/ec2-user/bootstrap.sh --only-show-errors
aws s3 cp s3://$B/ber/queue/jobrunner.sh /home/ec2-user/jobrunner.sh --only-show-errors
chown ec2-user /home/ec2-user/bootstrap.sh /home/ec2-user/jobrunner.sh /home/ec2-user/autostop.py
sudo -u ec2-user -i nohup bash /home/ec2-user/bootstrap.sh > /dev/null 2>&1 &
