#!/usr/bin/env bash
# Read-only. Run in AWS CloudShell of EACH account and paste the output in the team chat.
#   bash aws/00_account_check.sh
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh

echo "== identity"
aws sts get-caller-identity --output table

echo "== account plan (FREE plan blocks GPU instances; needs upgrade to PAID)"
aws freetier get-account-plan-state --output table 2>/dev/null || echo "  (plan API unavailable; check Billing console > Free Tier)"

echo "== GPU instance types offered in $AWS_REGION"
aws ec2 describe-instance-type-offerings --location-type region \
  --filters Name=instance-type,Values=g4dn.xlarge,g5.xlarge,g5.2xlarge,g6.xlarge,g6.2xlarge,g6e.xlarge \
  --query 'InstanceTypeOfferings[].InstanceType' --output text

echo "== EC2 GPU vCPU quotas"
for code in L-DB2E81BA L-3819A6DF; do
  aws service-quotas get-service-quota --service-code ec2 --quota-code "$code" \
    --query '[Quota.QuotaName, Quota.Value]' --output text
done

echo "== SageMaker GPU quotas (non-zero or relevant)"
aws service-quotas list-service-quotas --service-code sagemaker --max-items 5000 \
  --query "Quotas[?contains(QuotaName, 'ml.g4dn.xlarge') || contains(QuotaName, 'ml.g5.xlarge') || contains(QuotaName, 'ml.g5.2xlarge') || contains(QuotaName, 'ml.g6.xlarge')].[QuotaName, Value]" \
  --output text | sort

echo "== open quota requests"
aws service-quotas list-requested-service-quota-change-history --service-code sagemaker \
  --query "RequestedQuotas[?Status!='APPROVED' && Status!='DENIED'].[QuotaName, DesiredValue, Status]" --output text
aws service-quotas list-requested-service-quota-change-history --service-code ec2 \
  --query "RequestedQuotas[].[QuotaName, DesiredValue, Status]" --output text

echo "== anything running that costs money"
aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId, InstanceType, LaunchTime]' --output text
aws sagemaker list-notebook-instances --status-equals InService --query 'NotebookInstances[].NotebookInstanceName' --output text
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text
