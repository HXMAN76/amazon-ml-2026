#!/bin/bash
# Point the notebook scripts at your own S3 bucket. Usage: bash aws/notebook/set_bucket.sh <your-bucket-name>
# The scripts (bootstrap.sh, jobrunner.sh, onstart.sh) keep the bucket in one variable B; job scripts you write should sync code from s3://<bucket>/ber/code.
set -e
NEW=${1:?usage: set_bucket.sh <bucket>}
OLD=sagemaker-us-east-1-567503593043
sed -i "s/$OLD/$NEW/g" "$(dirname "$0")"/bootstrap.sh "$(dirname "$0")"/jobrunner.sh "$(dirname "$0")"/onstart.sh
grep -n "^B=" "$(dirname "$0")"/*.sh
