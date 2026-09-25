#!/bin/bash
# Polls s3://$B/<prefix>/pending/*.sh, runs each in the `ber` conda env, streams the log to <prefix>/live/ every 30 s while it runs
# and uploads the final log to <prefix>/done/.  Usage: jobrunner.sh [lane]   (default lane "" uses prefix "jobs"; lane "2" uses "jobs2").
# Several lanes run side by side; /tmp/job.lock exists while any lane runs a job so the idle auto-stop never kills a running job.
B=sagemaker-us-east-1-567503593043
LANE=${1:-}
PFX=jobs$LANE
LOCKS=/tmp/joblocks; mkdir -p $LOCKS
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
while true; do
  for k in $(aws s3 ls s3://$B/$PFX/pending/ | awk '{print $4}' | grep '\.sh$'); do
    n=${k%.sh}
    aws s3 mv s3://$B/$PFX/pending/$k /tmp/$k --only-show-errors || continue
    touch $LOCKS/$PFX /tmp/job.lock
    ( conda activate ber 2>/dev/null; cd /home/ec2-user/SageMaker; export PYTHONPATH=/home/ec2-user/SageMaker/ber/src; bash /tmp/$k ) > /tmp/$n.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      aws s3 cp /tmp/$n.log s3://$B/$PFX/live/$n.log --only-show-errors 2>/dev/null
      sleep 30
    done
    wait $pid; echo "exit=$?" >> /tmp/$n.log
    aws s3 cp /tmp/$n.log s3://$B/$PFX/done/$n.log --only-show-errors
    aws s3 rm s3://$B/$PFX/live/$n.log --only-show-errors 2>/dev/null
    rm -f $LOCKS/$PFX
    [ -z "$(ls -A $LOCKS 2>/dev/null)" ] && rm -f /tmp/job.lock
  done
  sleep 10
done
