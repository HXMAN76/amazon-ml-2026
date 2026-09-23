#!/usr/bin/env bash
# Cost alarm for EACH account. Emails at 25/50/75/90% of actual and 100% forecast.
#   ALERT_EMAIL=you@example.com bash aws/02_budget.sh
#
# IncludeCredit=false on purpose: with credits applied the net bill stays $0 and a normal
# budget would never fire until credits are gone. We want to see real usage burn.
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
: "${ALERT_EMAIL:?set ALERT_EMAIL=<your email>}"
ACCT=$(me)

cat > /tmp/budget.json <<EOF
{
  "BudgetName": "amlc-2026-usage",
  "BudgetLimit": {"Amount": "$BUDGET_USD", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostTypes": {"IncludeCredit": false, "IncludeRefund": false, "IncludeTax": true,
                "IncludeSubscription": true, "IncludeRecurring": true, "IncludeOtherSubscription": true,
                "IncludeSupport": true, "IncludeDiscount": true, "UseBlended": false}
}
EOF

notif() {  # type threshold
  printf '{"Notification":{"NotificationType":"%s","ComparisonOperator":"GREATER_THAN","Threshold":%s,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"%s"}]}' \
    "$1" "$2" "$ALERT_EMAIL"
}

if aws budgets describe-budget --account-id "$ACCT" --budget-name amlc-2026-usage >/dev/null 2>&1; then
  echo "budget exists; updating limit"
  aws budgets update-budget --account-id "$ACCT" --new-budget file:///tmp/budget.json
else
  aws budgets create-budget --account-id "$ACCT" --budget file:///tmp/budget.json \
    --notifications-with-subscribers "$(notif ACTUAL 25)" "$(notif ACTUAL 50)" "$(notif ACTUAL 75)" \
      "$(notif ACTUAL 90)" "$(notif FORECASTED 100)"
fi
echo "budget amlc-2026-usage = \$$BUDGET_USD/month -> $ALERT_EMAIL (account $ACCT)"
