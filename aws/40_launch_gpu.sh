#!/usr/bin/env bash
# Launch ONE GPU box only when a job needs it. Auto-terminates after MAX_HOURS no matter what.
# Needs: Paid plan + approved G/VT vCPU quota + 20_compute_role.sh run in this account.
#
#   INSTANCE_TYPE=g5.xlarge SPOT=1 MAX_HOURS=6 bash aws/40_launch_gpu.sh
#   aws ssm start-session --target <instance-id>        # shell, no SSH key / open port needed
#
# On the box: code is pulled from s3://$HUB_BUCKET/code/amlc.tar.gz (push it with `make push-code`).
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
SPOT="${SPOT:-1}"
MAX_HOURS="${MAX_HOURS:-6}"
DISK_GB="${DISK_GB:-200}"
NAME="${NAME:-amlc-gpu-$(whoami)}"

# AWS Deep Learning Base AMI (NVIDIA driver + CUDA, Ubuntu 22.04), latest, via public SSM parameter
AMI=$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
      --query Parameter.Value --output text)

cat > /tmp/userdata.sh <<EOF
#!/bin/bash
# hard cost cap: power off (=terminate, see shutdown behavior) after MAX_HOURS
shutdown -h +$((MAX_HOURS * 60))
sudo -u ubuntu bash -lc '
  curl -LsSf https://astral.sh/uv/install.sh | sh
  mkdir -p ~/amlc && cd ~/amlc
  aws s3 cp s3://$HUB_BUCKET/code/amlc.tar.gz - | tar xz || true
  echo "export AMLC_BUCKET=$HUB_BUCKET" >> ~/.bashrc
'
EOF

MARKET=()
[[ "$SPOT" == 1 ]] && MARKET=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')

ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1 \
  --iam-instance-profile Name="$COMPUTE_ROLE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --metadata-options HttpTokens=required \
  --user-data file:///tmp/userdata.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=amlc-2026}]" \
  "${MARKET[@]}" --query 'Instances[0].InstanceId' --output text)

echo "launched $ID ($INSTANCE_TYPE spot=$SPOT, terminates in ${MAX_HOURS}h, ${DISK_GB}GB gp3)"
echo "shell:     aws ssm start-session --target $ID"
echo "terminate: aws ec2 terminate-instances --instance-ids $ID"
