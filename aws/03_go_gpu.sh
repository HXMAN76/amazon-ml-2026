#!/usr/bin/env bash
# One paste to get an account ready for GPU access: upgrade to Paid, budget alert, GPU quota request.
# Run in the AWS CloudShell of each account that should get GPU access (us-east-1).
#   ALERT_EMAIL=you@example.com bash aws/03_go_gpu.sh
#
# Upgrading keeps the remaining credits. The card is charged only for usage beyond the credits,
# and the budget alert counts usage before credits, so its emails show real burn.
# Never join AWS Organizations: it forces the Paid plan on the account.
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh
: "${ALERT_EMAIL:?set ALERT_EMAIL=<your email>}"

plan=$(aws freetier get-account-plan-state --query accountPlanType --output text)
echo "account $(me)  plan=$plan  region=$AWS_REGION"

if [[ "$plan" == FREE ]]; then
  echo "Free plan blocks GPU instances. Upgrade to PAID keeps remaining credits."
  read -r -p "type UPGRADE to continue: " ans
  [[ "$ans" == UPGRADE ]] || { echo "aborted, nothing changed"; exit 1; }
  aws freetier upgrade-account-plan --account-plan-type PAID
fi

bash ./02_budget.sh
bash ./01_request_quotas.sh
echo
echo "Next: AWS reviews the quota case (minutes to days). Check: bash aws/00_account_check.sh"
