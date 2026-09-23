#!/usr/bin/env bash
# ACCOUNT A ONLY. Creates the shared data-hub bucket and grants accounts B/C/D access.
# No AWS Organizations needed (joining one would force Free plan accounts onto Paid plan).
#   bash aws/10_hub_bucket.sh
#
# Layout:
#   00-raw/          official dataset, immutable (only Account A can write; nobody can delete)
#   01-images/       downloaded images (+ manifest.jsonl)
#   02-processed/  03-features/  04-models/  05-predictions/  06-submissions/  07-experiments/
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
for v in ACCOUNT_A ACCOUNT_B ACCOUNT_C ACCOUNT_D; do : "${!v:?fill $v in aws/env.sh}"; done
[[ "$(me)" == "$ACCOUNT_A" ]] || { echo "run this in ACCOUNT_A ($ACCOUNT_A), you are in $(me)"; exit 1; }

if ! aws s3api head-bucket --bucket "$HUB_BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$HUB_BUCKET" --region "$AWS_REGION" \
    --create-bucket-configuration LocationConstraint="$AWS_REGION" \
    --object-ownership BucketOwnerEnforced
  echo "created s3://$HUB_BUCKET"
fi

aws s3api put-public-access-block --bucket "$HUB_BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Clean up failed multipart uploads (they are billed but invisible) and old scratch
aws s3api put-bucket-lifecycle-configuration --bucket "$HUB_BUCKET" --lifecycle-configuration '{
  "Rules": [
    {"ID": "abort-mpu", "Status": "Enabled", "Filter": {"Prefix": ""},
     "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2}},
    {"ID": "tmp-expire", "Status": "Enabled", "Filter": {"Prefix": "tmp/"}, "Expiration": {"Days": 3}}
  ]}'

TEAM="\"arn:aws:iam::$ACCOUNT_B:root\",\"arn:aws:iam::$ACCOUNT_C:root\",\"arn:aws:iam::$ACCOUNT_D:root\""
cat > /tmp/hub-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "TeamList", "Effect": "Allow", "Principal": {"AWS": [$TEAM]},
     "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": "arn:aws:s3:::$HUB_BUCKET"},
    {"Sid": "TeamRead", "Effect": "Allow", "Principal": {"AWS": [$TEAM]},
     "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::$HUB_BUCKET/*"},
    {"Sid": "TeamWriteDerived", "Effect": "Allow", "Principal": {"AWS": [$TEAM]},
     "Action": ["s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
     "NotResource": "arn:aws:s3:::$HUB_BUCKET/00-raw/*"},
    {"Sid": "RawIsImmutable", "Effect": "Deny", "Principal": "*",
     "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion"], "Resource": "arn:aws:s3:::$HUB_BUCKET/00-raw/*"},
    {"Sid": "TLSOnly", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
     "Resource": ["arn:aws:s3:::$HUB_BUCKET", "arn:aws:s3:::$HUB_BUCKET/*"],
     "Condition": {"Bool": {"aws:SecureTransport": "false"}}}
  ]
}
EOF
aws s3api put-bucket-policy --bucket "$HUB_BUCKET" --policy file:///tmp/hub-policy.json

for p in 00-raw 01-images 02-processed 03-features 04-models 05-predictions 06-submissions 07-experiments; do
  aws s3api put-object --bucket "$HUB_BUCKET" --key "$p/" >/dev/null
done
echo "hub ready: s3://$HUB_BUCKET  (B/C/D: $ACCOUNT_B $ACCOUNT_C $ACCOUNT_D)"
