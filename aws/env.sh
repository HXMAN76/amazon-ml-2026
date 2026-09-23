# Shared settings for every aws/*.sh script. Edit once, commit, everyone pulls.
# Source-safe: only variable definitions.

export AWS_REGION="${AWS_REGION:-ap-south-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"

# 12-digit account ids. Fill all four before running 10_hub_bucket.sh.
export ACCOUNT_A="${ACCOUNT_A:-}"   # data hub (S3, ECR)
export ACCOUNT_B="${ACCOUNT_B:-}"
export ACCOUNT_C="${ACCOUNT_C:-}"
export ACCOUNT_D="${ACCOUNT_D:-}"

export HUB_BUCKET="${HUB_BUCKET:-amlc-2026-hub-${ACCOUNT_A}}"
export ECR_REPO="${ECR_REPO:-amlc/base}"
export COMPUTE_ROLE="${COMPUTE_ROLE:-amlc-compute-role}"

# Budget per account (USD). Alerts fire on usage BEFORE credits are applied.
export BUDGET_USD="${BUDGET_USD:-180}"

me() { aws sts get-caller-identity --query Account --output text; }
