#!/bin/bash
set -euo pipefail

REGION="us-east-1"
ACCOUNT_ID="011015903780"
ECR_REPO="mlops-fraud-detection"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
LAMBDA_FUNCTION="mlops-fraud-detection-predict"
IMAGE_TAG="${1:-latest}"

push() {
    echo "=== Pushing Docker image to ECR ==="
    aws ecr get-login-password --region "${REGION}" | \
        docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

    docker build -t "${ECR_REPO}:${IMAGE_TAG}" .
    docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
    docker push "${ECR_URI}:${IMAGE_TAG}"
    echo "  Pushed: ${ECR_URI}:${IMAGE_TAG}"
}

deploy() {
    echo "=== Deploying to Lambda ==="
    aws lambda update-function-code \
        --function-name "${LAMBDA_FUNCTION}" \
        --image-uri "${ECR_URI}:${IMAGE_TAG}" \
        --region "${REGION}"

    echo "Waiting for Lambda to update..."
    aws lambda wait function-updated \
        --function-name "${LAMBDA_FUNCTION}" \
        --region "${REGION}"
    echo "  Lambda updated successfully"
}

case "${1:-help}" in
    push)   push ;;
    deploy) deploy ;;
    all)    push && deploy ;;
    *)
        echo "Usage: $0 {push|deploy|all} [image-tag]"
        echo "  push   - Build and push Docker image to ECR"
        echo "  deploy - Update Lambda function with latest image"
        echo "  all    - Push then deploy"
        exit 1
        ;;
esac
