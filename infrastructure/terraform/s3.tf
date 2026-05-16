resource "aws_s3_bucket" "mlops" {
  bucket = "${var.project_name}-${var.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlops" {
  bucket = aws_s3_bucket.mlops.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "mlops" {
  bucket = aws_s3_bucket.mlops.id

  rule {
    id     = "archive-old-models"
    status = "Enabled"
    filter {
      prefix = "models/"
    }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "expire-old-drift-reports"
    status = "Enabled"
    filter {
      prefix = "drift-reports/"
    }
    expiration {
      days = 365
    }
  }
}
