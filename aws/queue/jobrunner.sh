#!/bin/bash
# The team's runner (aws/notebook/jobrunner.sh) for this account: polls s3://$B/jobs/pending/*.sh, runs one job at a time (alphabetical)
# in the `ber` env from /home/ec2-user/SageMaker, streams the log to jobs/live/ every 30 s, uploads jobs/done/<name>.log ending with
# exit=<code>, holds /tmp/job.lock while a job runs (the idle auto-stop skips then), and writes jobs/runner.txt as a liveness beacon.
B=sagemaker-us-east-1-645311222213
SM=/home/ec2-user/SageMaker
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
while true; do
  echo "alive $(date -u '+%F %T') UTC" | aws s3 cp - s3://$B/jobs/runner.txt --only-show-errors 2>/dev/null
  for k in $(aws s3 ls s3://$B/jobs/pending/ | awk '{print $4}' | grep '\.sh$'); do
    n=${k%.sh}
    aws s3 mv s3://$B/jobs/pending/$k /tmp/$k --only-show-errors || continue
    touch /tmp/job.lock
    ( conda activate $SM/envs/ber 2>/dev/null; cd $SM; export BER_DATA=$SM/dataset BER_WORK=$SM/work PYTHONPATH=$SM/ber/src; bash /tmp/$k ) > /tmp/$n.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      aws s3 cp /tmp/$n.log s3://$B/jobs/live/$n.log --only-show-errors 2>/dev/null
      sleep 30
    done
    wait $pid; echo "exit=$?" >> /tmp/$n.log
    aws s3 cp /tmp/$n.log s3://$B/jobs/done/$n.log --only-show-errors
    aws s3 rm s3://$B/jobs/live/$n.log --only-show-errors 2>/dev/null
    rm -f /tmp/job.lock
  done
  sleep 10
done
