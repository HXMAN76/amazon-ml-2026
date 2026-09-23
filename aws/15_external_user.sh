#!/usr/bin/env bash
# ACCOUNT A ONLY. IAM user for non-AWS workers (Kaggle / Colab / teammates' laptops without
# their own AWS login) limited to the hub bucket. Keys go into Kaggle "Secrets", never into git.
#   bash aws/15_external_user.sh
#   aws iam create-access-key --user-name amlc-external    # run yourself; copy keys to Kaggle Secrets
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
[[ "$(me)" == "$ACCOUNT_A" ]] || { echo "run in ACCOUNT_A"; exit 1; }
USER=amlc-external

aws iam get-user --user-name "$USER" >/dev/null 2>&1 || aws iam create-user --user-name "$USER" >/dev/null
aws iam put-user-policy --user-name "$USER" --policy-name amlc-hub-only --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:ListBucket\", \"s3:GetBucketLocation\"],
     \"Resource\": \"arn:aws:s3:::$HUB_BUCKET\"},
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:AbortMultipartUpload\"],
     \"Resource\": \"arn:aws:s3:::$HUB_BUCKET/*\"}
  ]}"
echo "user $USER can read/write s3://$HUB_BUCKET only (no delete)."
echo "next: aws iam create-access-key --user-name $USER   -> add as Kaggle secrets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
