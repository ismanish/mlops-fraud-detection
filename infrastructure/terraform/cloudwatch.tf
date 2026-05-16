# Lambda log group
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project_name}-predict"
  retention_in_days = 30

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# API Gateway log group
resource "aws_cloudwatch_log_group" "api_gw" {
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 30

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Dashboard
resource "aws_cloudwatch_dashboard" "mlops" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Prediction Latency"
          metrics = [["MLOps/FraudDetection", "PredictionLatency"]]
          period  = 300
          stat    = "Average"
          region  = var.aws_region
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Fraud Detection Rate"
          metrics = [["MLOps/FraudDetection", "FraudPredicted"]]
          period  = 300
          stat    = "Sum"
          region  = var.aws_region
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Data Drift Detected"
          metrics = [["MLOps/FraudDetection", "DataDriftDetected"]]
          period  = 3600
          stat    = "Sum"
          region  = var.aws_region
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Model Degradation"
          metrics = [["MLOps/FraudDetection", "ModelDegraded"]]
          period  = 3600
          stat    = "Sum"
          region  = var.aws_region
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Lambda Invocations"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", "${var.project_name}-predict"],
            ["AWS/Lambda", "Errors", "FunctionName", "${var.project_name}-predict"]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Lambda Duration"
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", "${var.project_name}-predict"]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
        }
      }
    ]
  })
}

# Alarms
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Lambda error rate is too high"
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = "${var.project_name}-predict"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "drift_detected" {
  alarm_name          = "${var.project_name}-drift-detected"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "DataDriftDetected"
  namespace           = "MLOps/FraudDetection"
  period              = 3600
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Data drift detected — model may need retraining"
  treat_missing_data  = "notBreaching"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
