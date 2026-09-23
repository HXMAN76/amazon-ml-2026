#!/usr/bin/env bash
# Request small GPU quotas (quota != spend; nothing is launched).
# Run in CloudShell of EACH account.   DRY_RUN=1 bash aws/01_request_quotas.sh   to preview.
#
# NOTE: Free-plan accounts are usually denied GPU quota. Upgrade to Paid plan first
# (Billing console > Free Tier > Upgrade plan; remaining credits carry over).
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh
DRY_RUN="${DRY_RUN:-0}"

request() {  # service quota_code desired label
  local svc=$1 code=$2 want=$3 label=$4 cur
  cur=$(aws service-quotas get-service-quota --service-code "$svc" --quota-code "$code" \
          --query Quota.Value --output text 2>/dev/null || echo "?")
  if [[ "$cur" != "?" ]] && awk "BEGIN{exit !($cur >= $want)}"; then
    echo "OK      $label = $cur"
    return
  fi
  if [[ "$DRY_RUN" == 1 ]]; then echo "WOULD   $label: $cur -> $want"; return; fi
  if aws service-quotas request-service-quota-increase --service-code "$svc" --quota-code "$code" \
       --desired-value "$want" --query 'RequestedQuota.Status' --output text 2>/tmp/q.err; then
    echo "REQUEST $label: $cur -> $want"
  else
    echo "SKIP    $label: $(tail -1 /tmp/q.err)"
  fi
}

sm_code() {  # exact SageMaker quota name -> code
  aws service-quotas list-service-quotas --service-code sagemaker --max-items 5000 \
    --query "Quotas[?QuotaName=='$1'].QuotaCode | [0]" --output text
}

echo "== EC2 ($AWS_REGION): 8 vCPU = one g5/g6.2xlarge or two xlarge"
request ec2 L-DB2E81BA 8 "Running On-Demand G and VT instances (vCPU)"
request ec2 L-3819A6DF 8 "All G and VT Spot Instance Requests (vCPU)"

echo "== SageMaker ($AWS_REGION), 1 instance each"
# Priority order: training (incl. spot), processing (batch GPU jobs), transform. Endpoints not needed.
for itype in ml.g5.xlarge ml.g6.xlarge ml.g4dn.xlarge ml.g5.2xlarge; do
  for usage in "training job usage" "spot training job usage" "processing job usage" "transform job usage"; do
    name="$itype for $usage"
    code=$(sm_code "$name")
    if [[ -z "$code" || "$code" == "None" ]]; then echo "N/A     $name (not in region)"; continue; fi
    request sagemaker "$code" 1 "$name"
  done
done

echo
echo "Track: Service Quotas console > Quota request history. Re-run 00_account_check.sh tomorrow."
