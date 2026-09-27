#!/bin/bash
# The team runner (formerly aws/notebook/jobrunner.sh, see git history) for this account, with two lanes polled in parallel: s3://$B/jobs/ (GPU jobs) and
# s3://$B/jobs2/ (CPU jobs that may run at the same time). Each lane runs its pending *.sh one at a time (alphabetical) in the `ber` env from
# /home/ec2-user/SageMaker, streams the log to <lane>/live/ every 30 s and uploads <lane>/done/<name>.log ending with exit=<code>.
# /tmp/job.lock exists while any lane is busy (the idle auto-stop skips then); jobs/runner.txt is the liveness beacon.
B=sagemaker-us-east-1-645311222213
SM=/home/ec2-user/SageMaker
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh

lane() {
  local Q=$1
  while true; do
    for k in $(aws s3 ls s3://$B/$Q/pending/ | awk '{print $4}' | grep '\.sh$'); do
      n=${k%.sh}
      aws s3 mv s3://$B/$Q/pending/$k /tmp/$Q-$k --only-show-errors || continue
      touch /tmp/$Q.busy
      ( conda activate $SM/envs/ber 2>/dev/null; cd $SM; export BER_DATA=$SM/dataset BER_WORK=$SM/work PYTHONPATH=$SM/ber/src; bash /tmp/$Q-$k ) > /tmp/$Q-$n.log 2>&1 &
      pid=$!
      while kill -0 $pid 2>/dev/null; do
        aws s3 cp /tmp/$Q-$n.log s3://$B/$Q/live/$n.log --only-show-errors 2>/dev/null
        sleep 30
      done
      wait $pid; echo "exit=$?" >> /tmp/$Q-$n.log
      aws s3 cp /tmp/$Q-$n.log s3://$B/$Q/done/$n.log --only-show-errors
      aws s3 rm s3://$B/$Q/live/$n.log --only-show-errors 2>/dev/null
      rm -f /tmp/$Q.busy
    done
    sleep 10
  done
}

lane jobs &
lane jobs2 &
while true; do
  if ls /tmp/*.busy >/dev/null 2>&1; then touch /tmp/job.lock; else rm -f /tmp/job.lock; fi
  echo "alive $(date -u '+%F %T') UTC busy: $(ls /tmp/*.busy 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" | aws s3 cp - s3://$B/jobs/runner.txt --only-show-errors 2>/dev/null
  sleep 20
done
