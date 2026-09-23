#!/usr/bin/env bash
# EVERY account (A, B, C, D). Creates one role that EC2 instances and SageMaker jobs assume.
# It can read/write the hub bucket, pull from ECR, write logs. Creating a role costs nothing.
#   bash aws/20_compute_role.sh
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh
: "${ACCOUNT_A:?fill ACCOUNT_A in aws/env.sh}"

aws iam get-role --role-name "$COMPUTE_ROLE" >/dev/null 2>&1 || aws iam create-role \
  --role-name "$COMPUTE_ROLE" --description "amlc 2026 EC2 + SageMaker compute" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
      "Principal": {"Service": ["ec2.amazonaws.com", "sagemaker.amazonaws.com"]},
      "Action": "sts:AssumeRole"}]}' >/dev/null

cat > /tmp/compute-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "HubList", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
     "Resource": "arn:aws:s3:::$HUB_BUCKET"},
    {"Sid": "HubObjects", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
     "Resource": "arn:aws:s3:::$HUB_BUCKET/*"},
    {"Sid": "SageMakerDefaultBucket", "Effect": "Allow", "Action": "s3:*",
     "Resource": ["arn:aws:s3:::sagemaker-*", "arn:aws:s3:::sagemaker-*/*"]},
    {"Sid": "EcrPull", "Effect": "Allow",
     "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                "ecr:BatchCheckLayerAvailability"], "Resource": "*"},
    {"Sid": "Logs", "Effect": "Allow",
     "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
                "cloudwatch:PutMetricData"], "Resource": "*"}
  ]
}
EOF
aws iam put-role-policy --role-name "$COMPUTE_ROLE" --policy-name amlc-hub-access \
  --policy-document file:///tmp/compute-policy.json
# SSM lets you open a shell on EC2 without SSH keys / open ports
aws iam attach-role-policy --role-name "$COMPUTE_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

aws iam get-instance-profile --instance-profile-name "$COMPUTE_ROLE" >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name "$COMPUTE_ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$COMPUTE_ROLE" --role-name "$COMPUTE_ROLE"
}
echo "role + instance profile '$COMPUTE_ROLE' ready in account $(me)"
