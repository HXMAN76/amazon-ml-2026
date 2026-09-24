#!/bin/bash
# Polls s3://$B/jobs/pending/*.sh, runs each in the `ber` conda env, uploads log to jobs/done/.
# Keeps /tmp/job.lock while a job runs so the idle auto-stop does not kill long jobs.
B=sagemaker-us-east-1-567503593043
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
while true; do
  for k in $(aws s3 ls s3://$B/jobs/pending/ | awk '{print $4}' | grep '\.sh$'); do
    n=${k%.sh}
    aws s3 mv s3://$B/jobs/pending/$k /tmp/$k --only-show-errors || continue
    touch /tmp/job.lock
    ( conda activate ber 2>/dev/null; cd /home/ec2-user/SageMaker; export PYTHONPATH=/home/ec2-user/SageMaker/ber/src; bash /tmp/$k ) > /tmp/$n.log 2>&1
    echo "exit=$?" >> /tmp/$n.log
    aws s3 cp /tmp/$n.log s3://$B/jobs/done/$n.log --only-show-errors
    rm -f /tmp/job.lock
  done
  sleep 10
done
