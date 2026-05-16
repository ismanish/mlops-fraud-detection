output "s3_bucket_name" {
  description = "S3 bucket for data and model storage"
  value       = aws_s3_bucket.mlops.id
}

output "ecr_repository_url" {
  description = "ECR repository URL for Docker images"
  value       = aws_ecr_repository.mlops.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.prediction.function_name
}

output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${var.project_name}-dashboard"
}
