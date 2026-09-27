#!/bin/bash
# ec2-user, background: dataset + code onto the notebook volume, the `ber` env (kept ON the persistent volume, so it is built once,
# not at every start), then the job runner. The log goes to s3://$B/jobs/live/_bootstrap.log.
B=sagemaker-us-east-1-645311222213
SM=/home/ec2-user/SageMaker
ENV=$SM/envs/ber
L=$SM/queue/bootstrap.log
mkdir -p $SM/queue $SM/ber $SM/dataset $SM/work $SM/tools
exec >> $L 2>&1
up() { aws s3 cp $L s3://$B/jobs/live/_bootstrap.log --only-show-errors 2>/dev/null; }
echo "[$(date '+%F %T')] bootstrap start; identity $(aws sts get-caller-identity --query Arn --output text)"; up
aws s3 sync s3://$B/ber/dataset/ $SM/dataset/ --only-show-errors
aws s3 sync s3://$B/ber/code $SM/ber --delete --exclude 'work/*' --exact-timestamps --only-show-errors
aws s3 sync s3://$B/ber/tools $SM/tools --exact-timestamps --only-show-errors
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
[ -x $ENV/bin/python ] || conda create -y -q -p $ENV python=3.12
conda activate $ENV
python -m pip install -q -r $SM/ber/requirements.txt \
  && python -c "import polars, duckdb, xgboost, rapidfuzz, torch; print('env ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  && echo ready > $SM/queue/.env_ready
echo "[$(date '+%F %T')] env done: $(cat $SM/queue/.env_ready 2>/dev/null || echo NOT READY)"; up
pgrep -f jobrunner.sh >/dev/null || nohup bash /home/ec2-user/jobrunner.sh > $SM/queue/jobrunner.log 2>&1 &
echo "[$(date '+%F %T')] runner started"; up
