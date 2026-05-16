variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mlops-fraud-detection"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
  default     = "011015903780"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}
