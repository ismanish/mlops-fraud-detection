#!/bin/bash
set -euo pipefail

REGION="us-east-1"
ACCOUNT_ID="011015903780"
BUCKET="mlops-fraud-detection-${ACCOUNT_ID}"
ECR_REPO="mlops-fraud-detection"

echo "=== Setting up AWS resources for MLOps project ==="

# 1. Create S3 bucket
echo "Creating S3 bucket: ${BUCKET}"
if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
    echo "  Bucket already exists"
else
    aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}"
    aws s3api put-bucket-versioning --bucket "${BUCKET}" \
        --versioning-configuration Status=Enabled
    aws s3api put-public-access-block --bucket "${BUCKET}" \
        --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo "  Bucket created with versioning and public access blocked"
fi

# 2. Create ECR repository
echo "Creating ECR repository: ${ECR_REPO}"
if aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" 2>/dev/null; then
    echo "  Repository already exists"
else
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=true
    echo "  Repository created"
fi

# 3. Upload folder structure to S3
echo "Creating S3 folder structure..."
for prefix in data/raw data/processed models metrics drift-reports; do
    aws s3api put-object --bucket "${BUCKET}" --key "${prefix}/" --region "${REGION}" > /dev/null
done
echo "  S3 folder structure created"

echo ""
echo "=== AWS Setup Complete ==="
echo "S3 Bucket: ${BUCKET}"
echo "ECR Repo:  ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
