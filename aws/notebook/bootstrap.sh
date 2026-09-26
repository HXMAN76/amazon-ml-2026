#!/bin/bash
# Runs as ec2-user: sync code, build the `ber` env, start the job runners (lane 1 = prefix jobs, lane 2 = prefix jobs2). Idempotent.
B=sagemaker-us-east-1-567503593043
mkdir -p /home/ec2-user/SageMaker/ber /home/ec2-user/SageMaker/dataset
aws s3 sync s3://$B/ber/code /home/ec2-user/SageMaker/ber --only-show-errors
aws s3 cp s3://$B/ber/jobrunner.sh /home/ec2-user/jobrunner.sh --only-show-errors
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q '^ber ' || conda create -y -q -n ber python=3.12
conda activate ber
pip install -q -r /home/ec2-user/SageMaker/ber/requirements.txt && python -c "import pandas,scipy,sklearn,lightgbm,rapidfuzz" && echo ready > /home/ec2-user/SageMaker/ber/.env_ready
# job lanes: the main notebook uses prefixes jobs, jobs2, jobs3, jobs4; the second notebook (test-notebook-2) uses jobsB, jobsB2, jobsB3, jobsB4
NBNAME=$(python3 -c "import json;print(json.load(open('/opt/ml/metadata/resource-metadata.json'))['ResourceName'])" 2>/dev/null || echo test-notebook)
if [ "$NBNAME" = "test-notebook-2" ]; then LANES="B B2 B3 B4"; else LANES="_ 2 3 4"; fi
for L in $LANES; do
  A=$L; [ "$L" = "_" ] && A=""
  pgrep -f "jobrunner.sh $A\$" >/dev/null || nohup bash /home/ec2-user/jobrunner.sh $A > /home/ec2-user/jobrunner_$L.log 2>&1 &
done
# diagnostic heartbeat: a snapshot of the machine state to S3 every minute (the last one before a stall shows what was happening)
cat > /home/ec2-user/diag.sh <<'D'
#!/bin/bash
B=sagemaker-us-east-1-567503593043
NB=$(python3 -c "import json;print(json.load(open('/opt/ml/metadata/resource-metadata.json'))['ResourceName'])" 2>/dev/null || echo test-notebook)
while true; do
  { date; uptime; free -m | head -2; df -h / /home/ec2-user/SageMaker /tmp | tail -3
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | head -2
    ps aux --sort=-%mem | head -6 | cut -c1-160; dmesg 2>/dev/null | tail -8; } > /tmp/diag.txt 2>&1
  timeout 20 aws s3 cp /tmp/diag.txt s3://$B/diag/$NB.txt --only-show-errors
  cp /tmp/diag.txt /home/ec2-user/diag_$(date +%H%M).txt 2>/dev/null
  sleep 60
done
D
pgrep -f "diag.sh" >/dev/null || nohup setsid bash /home/ec2-user/diag.sh > /dev/null 2>&1 &
