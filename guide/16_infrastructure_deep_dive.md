# Chapter 16: Infrastructure Deep Dive

## Every Terraform Resource, Every Dockerfile Line, Every Shell Script Explained

This guide dissects the entire infrastructure layer of our MLOps project. We cover
Terraform for AWS resource provisioning, Docker for containerization, and shell scripts
for deployment automation. By the end, you will understand not just WHAT each line does,
but WHY it exists and what would go wrong without it.

---

## 1. Terraform Fundamentals

### 1.1 What is Infrastructure as Code?

Infrastructure as Code (IaC) means defining your cloud resources in text files (code)
instead of clicking through the AWS Console. Your infrastructure becomes:

- **Versionable** -- Track changes in Git. See who changed what, when, and why.
- **Repeatable** -- Destroy and recreate the entire environment with one command.
- **Reviewable** -- Pull requests for infrastructure changes, just like code changes.
- **Testable** -- Validate configurations before applying them.
- **Documented** -- The code IS the documentation. No more "tribal knowledge."

**Without IaC:** "Hey, who created this S3 bucket? What are its settings? Why does this
Lambda function have admin permissions?" Nobody knows. The person who set it up left the
company.

**With IaC:** Read `s3.tf`. Every setting is documented. The Git history shows every
change and who made it.

### 1.2 Terraform vs CloudFormation vs Pulumi

| Feature            | Terraform              | CloudFormation         | Pulumi                 |
|--------------------|------------------------|------------------------|------------------------|
| Language           | HCL (HashiCorp Config) | JSON/YAML              | Python, TypeScript, Go |
| Cloud Support      | Multi-cloud            | AWS only               | Multi-cloud            |
| State Management   | State file (you manage)| AWS manages state      | State file or cloud    |
| Learning Curve     | Medium                 | Medium                 | Low (if you know the language) |
| Community          | Largest                | AWS-focused            | Growing                |
| Drift Detection    | `terraform plan`       | Drift detection feature| `pulumi preview`       |
| Cost               | Free (open source)     | Free (AWS service)     | Free tier + paid       |

**Real tradeoffs:**
- **Terraform** wins for multi-cloud or if the team already knows HCL. Most MLOps teams
  choose Terraform because ML infrastructure often spans AWS (compute) + GCP (BigQuery) +
  on-prem (GPUs).
- **CloudFormation** wins if you are 100% AWS and want AWS to manage state.
- **Pulumi** wins if the team hates learning a new language (write infra in Python).

We use Terraform because it is the industry standard and the most commonly asked about
in interviews.

### 1.3 The Terraform Lifecycle

```
terraform init  -->  terraform plan  -->  terraform apply  -->  terraform destroy
```

**`terraform init`** -- Downloads provider plugins (e.g., the AWS provider) and
initializes the working directory. You run this once when starting, or after adding a new
provider. It creates a `.terraform/` directory with cached plugins.

**`terraform plan`** -- Shows what Terraform WOULD do without actually doing it. Reads
your `.tf` files, compares them to the current state, and outputs a diff:

```
+ aws_s3_bucket.mlops will be created
~ aws_lambda_function.prediction will be updated in-place
- aws_cloudwatch_metric_alarm.old will be destroyed
```

This is like `git diff` for infrastructure. Always review the plan before applying.

**`terraform apply`** -- Executes the plan. Creates, updates, or destroys resources to
match your configuration. Terraform handles the correct ORDER of operations (create the
S3 bucket before the Lambda that needs it).

**`terraform destroy`** -- Deletes ALL resources managed by Terraform. Dangerous but
useful for tearing down development environments.

### 1.4 State File

Terraform maintains a **state file** (`terraform.tfstate`) that records:
- Every resource Terraform manages
- The resource's current configuration
- Metadata (resource IDs, ARNs, etc.)

**Why does it matter?**

1. **Terraform compares state to config** to determine what changed. Without state,
   Terraform would not know that `aws_s3_bucket.mlops` already exists and would try to
   create a duplicate (which would fail because S3 bucket names are globally unique).

2. **State contains sensitive data** -- database passwords, access keys, etc. NEVER
   commit `terraform.tfstate` to Git. Store it in S3 with encryption.

3. **State locking** prevents two people from running `terraform apply` simultaneously
   (which could corrupt state). Use DynamoDB for state locking with S3 backend.

### 1.5 Version Pinning

```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
```

- **`required_version = ">= 1.5.0"`** -- Terraform CLI must be at least version 1.5.0.
  Prevents running with an old Terraform version that might not support features used in
  these config files.

- **`version = "~> 5.0"`** -- The `~>` operator means "pessimistic constraint." It allows
  `5.0`, `5.1`, `5.99` but NOT `6.0`. This means you get bug fixes and new features
  within the 5.x line, but not breaking changes from a major version bump.

### 1.6 Real Example: Recreating Infrastructure After Accidental Deletion

Someone accidentally deletes the S3 bucket through the AWS Console. Panic ensues. With
Terraform:

```bash
terraform plan
# Output: aws_s3_bucket.mlops will be created (it notices the bucket is missing)

terraform apply
# Bucket recreated with all the correct settings (versioning, encryption, etc.)
```

The entire infrastructure is back in 2 minutes. Without Terraform, someone would need to
remember all the settings and manually reconfigure everything.

---

## 2. Every Terraform Resource -- Line by Line

### 2.1 variables.tf -- Input Parameters

```hcl
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
```

**What are Terraform variables?** Variables let you parameterize your configuration.
Instead of hardcoding `us-east-1` in 15 places, you define it once as a variable and
reference `var.aws_region` everywhere.

**Each variable has three parts:**

1. **`description`** -- Human-readable explanation. Shows up in `terraform plan` output
   and documentation generators.
2. **`type`** -- The data type. Terraform supports `string`, `number`, `bool`, `list`,
   `map`, `object`, and `set`. Type checking prevents passing a number where a string is
   expected.
3. **`default`** -- The value used if not explicitly overridden. If there is no default,
   Terraform prompts you for the value (or errors if running non-interactively).

**`account_id`** is included as a variable because AWS account IDs are needed for
constructing S3 bucket names (global uniqueness) and ECR URIs. It is a string, not a
number, because account IDs can have leading zeros.

**Override at apply time:**

```bash
terraform apply -var="environment=staging" -var="aws_region=eu-west-1"
```

Or create a `terraform.tfvars` file:

```hcl
environment = "staging"
aws_region  = "eu-west-1"
```

### 2.2 main.tf -- Provider Configuration

```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

**The `terraform` block** declares requirements for Terraform itself and its providers.

**`source = "hashicorp/aws"`** tells Terraform to download the AWS provider from the
Terraform Registry (registry.terraform.io). The format is `namespace/type`. `hashicorp`
is the publisher (HashiCorp, the company behind Terraform). `aws` is the provider type.

**The `provider` block** configures the AWS provider. `region = var.aws_region` tells the
provider to create all resources in the specified region (us-east-1 by default).

The provider handles authentication automatically by looking for credentials in this
order:
1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. Shared credentials file (`~/.aws/credentials`)
3. IAM role (if running on an EC2 instance)
4. AWS SSO (if configured)

### 2.3 s3.tf -- S3 Bucket Configuration

#### The Bucket

```hcl
resource "aws_s3_bucket" "mlops" {
  bucket = "${var.project_name}-${var.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

**`resource "aws_s3_bucket" "mlops"`** -- Two names:
- `aws_s3_bucket` is the resource TYPE (from the AWS provider)
- `mlops` is the local NAME (used to reference this resource elsewhere in Terraform)

**`bucket = "${var.project_name}-${var.account_id}"`** constructs the bucket name:
`mlops-fraud-detection-011015903780`. The account ID is appended because S3 bucket names
must be **globally unique** across ALL AWS accounts worldwide. Without the account ID,
someone else could have already taken `mlops-fraud-detection`.

**Tags:** Every resource gets three tags:
- `Project` -- For cost allocation. AWS Cost Explorer can group costs by tag, so you see
  "the fraud-detection project costs $X/month."
- `Environment` -- Distinguishes production from staging resources.
- `ManagedBy = "terraform"` -- Signals to humans that this resource is managed by code.
  DO NOT modify it in the AWS Console (your changes will be overwritten by the next
  `terraform apply`).

#### Versioning

```hcl
resource "aws_s3_bucket_versioning" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  versioning_configuration {
    status = "Enabled"
  }
}
```

**Why version S3 objects?** Every time you upload a file with the same key (path), S3
keeps the old version. This is critical for ML:

- **Model rollback**: You upload `models/model.pkl` with a new version. The new model
  performs worse in production. With versioning, you can restore the previous version in
  seconds.
- **Data lineage**: You can trace which version of the training data produced which model.
- **Accidental deletion recovery**: If someone deletes a file, the previous version still
  exists (marked with a delete marker).

`aws_s3_bucket.mlops.id` is a **resource reference**. Terraform knows this versioning
resource depends on the bucket and will create the bucket first.

#### Encryption

```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

**Encryption at rest** means data is encrypted when stored on disk. Even if someone gains
physical access to AWS's storage hardware, they cannot read your data.

**`AES256`** (also called SSE-S3) uses AWS-managed encryption keys. It is free and
requires no key management from you. For stricter compliance, you could use
`aws:kms` with customer-managed keys, but that costs $1/month per key.

Why does this matter for ML? Training data may contain sensitive information (customer
transactions, personal data). Encryption is often REQUIRED by regulations like GDPR,
HIPAA, and PCI-DSS.

#### Public Access Block

```hcl
resource "aws_s3_bucket_public_access_block" "mlops" {
  bucket = aws_s3_bucket.mlops.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

This is the **nuclear option** for preventing public access. All four settings set to
`true`:

- **`block_public_acls`** -- Rejects any PUT request that includes a public ACL. Prevents
  uploading a file with public read access.
- **`block_public_policy`** -- Rejects any bucket policy that grants public access.
- **`ignore_public_acls`** -- Even if a public ACL exists, S3 ignores it.
- **`restrict_public_buckets`** -- Restricts access to bucket owner and AWS services only.

**Why all four?** Defense in depth. Each setting blocks a different way someone could
accidentally make the bucket public. Data breaches from misconfigured S3 buckets are
one of the most common security incidents in AWS. Companies have leaked millions of
records because of a single misconfigured bucket policy.

#### Lifecycle Rules

```hcl
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
```

**Two lifecycle rules for cost optimization:**

**Rule 1: Archive old models** -- Model files in `models/` are moved to
**STANDARD_IA** (Infrequent Access) after 90 days. Standard IA costs 45% less than
Standard storage but has a per-retrieval fee. Old models are rarely accessed but must be
kept for audit/rollback purposes. After 90 days, the cost savings justify the retrieval
fee.

**Rule 2: Expire old drift reports** -- Drift reports older than 365 days are
**deleted**. Unlike models (which may be needed for regulatory audits), drift reports are
operational data that loses value over time. Deleting them after a year keeps storage
costs down.

**`filter { prefix = "models/" }`** means this rule only applies to objects whose key
starts with `models/`. Objects in `data/` or `metrics/` are unaffected.

### 2.4 ecr.tf -- Container Registry

```hcl
resource "aws_ecr_repository" "mlops" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

**What is ECR?** Amazon Elastic Container Registry is a Docker image repository. Think of
it as a private Docker Hub hosted by AWS. You push Docker images to ECR, and AWS services
(Lambda, ECS, EKS) pull images from ECR.

**`image_tag_mutability = "MUTABLE"`** -- Allows overwriting a tag. When you push a new
image with tag `latest`, it replaces the old `latest`. If set to `IMMUTABLE`, pushing to
an existing tag would be rejected. MUTABLE is convenient for development but less safe for
production (someone could overwrite a production tag).

**Interview note:** In strict production environments, you use `IMMUTABLE` tags and always
deploy by SHA digest, not by tag name.

**`scan_on_push = true`** -- Every pushed image is automatically scanned for known
vulnerabilities (CVEs). If your base image has a critical security vulnerability, ECR
flags it. This is crucial for production ML services that handle sensitive data.

#### Lifecycle Policy

```hcl
resource "aws_ecr_lifecycle_policy" "mlops" {
  repository = aws_ecr_repository.mlops.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
```

**Why keep only 10 images?** Docker images are large (often 500MB-2GB for ML images).
Each deployment pushes a new image. After 50 deployments, you would have 50-100 GB of
images stored in ECR, costing money.

The lifecycle policy automatically deletes the oldest images when you exceed 10. This
keeps the 10 most recent images (enough for rollback) and saves storage costs.

**`tagStatus = "any"`** means it counts both tagged and untagged images.
**`countType = "imageCountMoreThan"` with `countNumber = 10`** means "when there are more
than 10 images, delete the extras."
**`rulePriority = 1`** -- Rules are evaluated in priority order. With only one rule, the
priority does not matter, but it is required.

### 2.5 lambda.tf -- The Core of Our Serving Infrastructure

This is the largest and most complex Terraform file. It defines the IAM role, Lambda
function, API Gateway, and all the connections between them.

#### IAM Role -- The Identity for Lambda

```hcl
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

**What is an IAM Role?** An IAM (Identity and Access Management) role is a set of
permissions that an AWS service can "assume" (take on). Think of it as a uniform: when
Lambda puts on this role, it can do everything the role allows.

**The Trust Policy (`assume_role_policy`)** answers: "WHO can assume this role?" The
answer is `lambda.amazonaws.com` -- only the AWS Lambda service. No human user, no EC2
instance, no other service can use this role. This is a security boundary.

**`sts:AssumeRole`** is the AWS Security Token Service action that allows assuming a role.
When Lambda starts your function, it calls STS to get temporary credentials for this role.

**Why `jsonencode()` instead of a JSON string?** Terraform's `jsonencode()` function
converts a Terraform map to valid JSON. It is less error-prone than writing JSON as a
string (no escaping quotes, Terraform validates the structure).

#### Basic Execution Policy

```hcl
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
```

This attaches an AWS **managed policy** to our role. `AWSLambdaBasicExecutionRole` grants
permission to write logs to CloudWatch. Every Lambda function needs this to produce logs.

**Managed policies** are pre-built by AWS and maintained by AWS. You cannot edit them,
but they cover common use cases. Using them is preferred over writing custom policies
for standard permissions.

#### Custom S3 and CloudWatch Policy

```hcl
resource "aws_iam_role_policy" "lambda_s3_cloudwatch" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlops.arn,
          "${aws_s3_bucket.mlops.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}
```

This is an **inline policy** -- custom permissions written directly in the Terraform
config. It grants two sets of permissions:

**Statement 1: S3 Access**
- `s3:GetObject` -- Read files from S3 (loading the model at startup)
- `s3:PutObject` -- Write files to S3 (saving predictions, drift reports)
- `s3:ListBucket` -- List files in the bucket (finding the latest model)

The `Resource` array specifies TWO ARNs:
- `aws_s3_bucket.mlops.arn` -- The bucket itself (needed for `ListBucket`)
- `"${aws_s3_bucket.mlops.arn}/*"` -- All objects IN the bucket (needed for Get/Put)

This is a common S3 permission gotcha: bucket-level and object-level permissions are
separate.

**Statement 2: CloudWatch Access**
- `cloudwatch:PutMetricData` -- Publish custom metrics (prediction latency, fraud rate)
- `Resource = "*"` -- CloudWatch does not support resource-level permissions for
  PutMetricData. This is an AWS limitation, not a security oversight.

**The Principle of Least Privilege:** This Lambda function can ONLY read/write to its
specific S3 bucket and publish CloudWatch metrics. It cannot access other S3 buckets,
modify IAM roles, launch EC2 instances, or do anything else. If the function is
compromised, the blast radius is contained.

#### Lambda Function

```hcl
resource "aws_lambda_function" "prediction" {
  function_name = "${var.project_name}-predict"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.mlops.repository_url}:latest"
  timeout       = 60
  memory_size   = 512

  environment {
    variables = {
      ENVIRONMENT = var.environment
      S3_BUCKET   = aws_s3_bucket.mlops.id
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  depends_on = [aws_ecr_repository.mlops]
}
```

**Line-by-line:**

- **`function_name`** -- The name visible in the AWS Console and used in CLI commands.
  `mlops-fraud-detection-predict`.

- **`role`** -- Which IAM role this function assumes. References the role we created above.

- **`package_type = "Image"`** -- This function runs from a Docker container (not a ZIP
  file). Container-based Lambda supports images up to 10 GB and lets you include custom
  binaries, system libraries, and ML model files.

- **`image_uri`** -- The Docker image to run. Points to our ECR repository with the
  `latest` tag. On initial Terraform apply, this might fail if no image has been pushed
  yet. That is expected -- push an image first, then apply.

- **`timeout = 60`** -- The function has 60 seconds to complete. If it takes longer,
  Lambda kills it. ML inference should be fast (< 1 second), but the first invocation
  (cold start) loads the model from disk, which can take 10-30 seconds for large models.
  The 60-second timeout provides headroom for cold starts.

- **`memory_size = 512`** -- 512 MB of RAM. Lambda allocates CPU proportionally to memory.
  512 MB gives ~0.3 vCPU, which is enough for sklearn inference. For deep learning
  inference, you would need 1024-3008 MB.

- **`environment.variables`** -- Environment variables injected into the container.
  `ENVIRONMENT` tells the code whether it is running in production or staging.
  `S3_BUCKET` provides the bucket name so the code does not need to hardcode it.

- **`depends_on = [aws_ecr_repository.mlops]`** -- Explicit dependency. Terraform must
  create the ECR repository before creating the Lambda function (because the function
  references the repository URL). Terraform usually infers dependencies from resource
  references, but `depends_on` makes it explicit for clarity.

#### API Gateway

```hcl
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_headers = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_origins = ["*"]
    max_age       = 86400
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

**API Gateway** is the front door to your Lambda function. It:
1. Accepts HTTP requests from the internet
2. Routes them to the correct Lambda function
3. Returns the Lambda's response to the client

**`protocol_type = "HTTP"`** -- HTTP API (v2), not REST API (v1). HTTP APIs are cheaper
(~70% less), faster (~60% lower latency), and simpler. REST APIs offer more features
(request validation, API keys, usage plans) but we do not need them.

**CORS Configuration:**
- `allow_headers = ["*"]` -- Allow any HTTP header
- `allow_methods = ["GET", "POST", "OPTIONS"]` -- Allow these HTTP methods
- `allow_origins = ["*"]` -- Allow requests from any domain
- `max_age = 86400` -- Cache CORS preflight responses for 24 hours (86400 seconds)

**Why CORS?** If a web application (running in a browser) tries to call our API, the
browser's Same-Origin Policy would block it unless CORS headers are present. `allow_origins = ["*"]` is permissive (allows any website). In production, you would
restrict this to your specific domains.

#### API Gateway Stage

```hcl
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
      latency        = "$context.integrationLatency"
    })
  }
}
```

**What is a stage?** A stage is a deployment snapshot of your API. You could have
`staging` and `production` stages with different configurations. `$default` is a special
stage name that handles requests without a stage prefix in the URL.

**`auto_deploy = true`** -- Any route or integration changes are automatically deployed.
Without this, you would need to explicitly deploy changes (like a manual promotion step).

**Access log format** -- Every API request is logged to CloudWatch with these fields:
- `requestId` -- Unique ID for the request (for tracing)
- `ip` -- Client IP address
- `requestTime` -- When the request arrived
- `httpMethod` -- GET, POST, etc.
- `routeKey` -- Which route was matched (POST /predict, GET /health)
- `status` -- HTTP status code (200, 400, 500)
- `responseLength` -- Size of the response body
- `latency` -- How long Lambda took to respond (in milliseconds)

These logs are invaluable for debugging production issues: "The /predict endpoint returned
500 errors from IP 1.2.3.4 at 3:45 PM with 2500ms latency."

#### API Gateway Integration

```hcl
resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.prediction.invoke_arn
  payload_format_version = "2.0"
}
```

This connects API Gateway to the Lambda function.

- **`integration_type = "AWS_PROXY"`** -- API Gateway passes the entire HTTP request to
  Lambda as a JSON event and returns Lambda's response directly. No request/response
  transformation. This is the most common integration type.

- **`integration_uri`** -- The Lambda function's invoke ARN (the address API Gateway uses
  to call Lambda).

- **`payload_format_version = "2.0"`** -- The format of the event object Lambda receives.
  Version 2.0 is simpler and includes the request body, headers, query parameters, and
  path parameters in a cleaner structure than version 1.0.

#### Routes

```hcl
resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}
```

Two routes, both pointing to the same Lambda function:

- **`POST /predict`** -- The prediction endpoint. Clients send transaction features in the
  request body and receive a fraud probability.
- **`GET /health`** -- The health check endpoint. Returns 200 if the service is up. Used
  by the deploy workflow's smoke test and by monitoring systems.

The Lambda function's code reads the HTTP method and path to determine which handler to
invoke internally.

#### Lambda Permission

```hcl
resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.prediction.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
```

This is a **resource-based policy** on the Lambda function. It says: "Allow the API
Gateway service (`apigateway.amazonaws.com`) to invoke this Lambda function."

Without this permission, API Gateway would receive an `AccessDeniedException` when trying
to call Lambda, and every API request would return a 500 error.

**`source_arn = "${aws_apigatewayv2_api.api.execution_arn}/*/*"`** -- The `/*/*` means
"any stage, any route." This allows both the `/predict` and `/health` routes from any
stage to invoke the function.

### 2.6 cloudwatch.tf -- Monitoring Infrastructure

#### Log Groups

```hcl
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project_name}-predict"
  retention_in_days = 30

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_cloudwatch_log_group" "api_gw" {
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 30

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

Two log groups:
1. **Lambda logs** (`/aws/lambda/...`) -- All print statements, errors, and structured
   logs from the Lambda function.
2. **API Gateway logs** (`/aws/apigateway/...`) -- Access logs for every API request.

**`retention_in_days = 30`** -- Logs older than 30 days are automatically deleted. Without
this, logs accumulate forever. CloudWatch charges $0.50/GB for ingestion and $0.03/GB/month
for storage. A busy ML API generating 10 GB of logs per month would cost $3.60/year in
storage without retention -- not much, but it adds up across services. More importantly,
30 days is enough for debugging recent issues, and older logs rarely provide value.

#### Dashboard

```hcl
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
      ...
    ]
  })
}
```

The dashboard is defined as a **JSON layout**. Each widget has:

- **`x`, `y`** -- Grid position (0,0 is top-left). The grid is 24 units wide.
- **`width`, `height`** -- Size in grid units. `width=12` takes half the page.
- **`metrics`** -- The metric to display. Format: `[namespace, metric_name]`.
  `MLOps/FraudDetection` is our custom namespace. `PredictionLatency` is a metric our
  application code publishes.
- **`period = 300`** -- Aggregate data in 5-minute (300-second) windows.
- **`stat = "Average"`** -- Show the average value in each period.

**The six dashboard widgets:**

| Widget             | Metric                  | What It Shows                              |
|--------------------|-------------------------|--------------------------------------------|
| Prediction Latency | PredictionLatency       | Average time per prediction (ms)           |
| Fraud Detection Rate | FraudPredicted        | Number of fraud predictions per period     |
| Data Drift Detected | DataDriftDetected      | Count of drift detection events            |
| Model Degradation  | ModelDegraded           | Whether the model's performance is dropping|
| Lambda Invocations | Invocations + Errors    | Request count and error count              |
| Lambda Duration    | Duration                | How long Lambda takes per invocation       |

The first four are **custom metrics** (published by our code). The last two are **AWS
built-in metrics** (published automatically by Lambda).

#### Alarms

```hcl
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
```

**How this alarm works:**

1. Every 5 minutes (`period = 300`), CloudWatch counts the total Lambda errors (`Sum`).
2. If the count exceeds 10 (`threshold = 10`)...
3. ...for 2 consecutive periods (`evaluation_periods = 2`)...
4. ...the alarm enters the ALARM state.

**Why 2 evaluation periods?** To avoid false alarms. A single period with 11 errors
might be a transient spike (a burst of malformed requests). Two consecutive periods with
10+ errors suggests a real problem.

**`treat_missing_data = "notBreaching"`** -- If there is no data for a period (the Lambda
function was not invoked), treat it as "everything is fine." Without this, a period with
no invocations would be treated as "insufficient data," which could prevent the alarm from
clearing after an incident resolves.

This matters for ML APIs that have low traffic during off-hours. At 3 AM, the function
might receive zero requests. You do not want the alarm to be in an ambiguous state just
because no one is using the API.

```hcl
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
```

**The drift alarm** fires if ANY drift detection event occurs in a 1-hour window
(`threshold = 0`, `GreaterThanThreshold`). Only 1 evaluation period because drift is
always significant -- even a single detection should trigger investigation.

This alarm could be connected to an SNS topic that sends email/Slack notifications, or
it could trigger an automated retraining pipeline via an EventBridge rule.

### 2.7 outputs.tf -- Exposing Values

```hcl
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
```

Outputs are values that Terraform prints after `terraform apply` and stores in state.
They serve three purposes:

1. **Human consumption** -- After apply, you see the API endpoint URL and can test it
   immediately.
2. **Script consumption** -- Shell scripts can read outputs:
   `terraform output -raw api_endpoint` returns just the URL.
3. **Module consumption** -- If this Terraform config is used as a module by another
   config, the parent can reference these outputs.

The `cloudwatch_dashboard_url` is particularly helpful -- it constructs a direct link to
the monitoring dashboard so you do not have to navigate through the AWS Console.

---

## 3. Dockerfile -- Line by Line

```dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY params.yaml ${LAMBDA_TASK_ROOT}/

COPY models/ ${LAMBDA_TASK_ROOT}/models/

CMD ["src.serving.lambda_handler.handler"]
```

### Line 1: `FROM public.ecr.aws/lambda/python:3.12`

**The base image.** This is AWS's official Lambda runtime image for Python 3.12, hosted
on the public ECR gallery.

**Why not the standard Python image (`python:3.12`)?** The Lambda runtime image includes:
- The Lambda Runtime Interface Client (RIC) -- handles communication with the Lambda
  service
- The Lambda Runtime Interface Emulator (RIE) -- allows local testing
- Correct directory structure (`/var/task` for your code)
- Optimized for Lambda's execution environment

A standard Python image would not work with Lambda's container runtime without
significant additional configuration.

**`${LAMBDA_TASK_ROOT}`** is an environment variable set by the base image. It points to
`/var/task` -- the directory where Lambda expects your code.

### Lines 3-4: Install Dependencies

```dockerfile
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt
```

**Why copy `requirements.txt` first, BEFORE the source code?** Docker layer caching.

Docker builds images in layers. Each instruction creates a layer. If a layer's input has
not changed, Docker reuses the cached layer instead of rebuilding it.

By copying `requirements.txt` first:
- If only your source code changes (not dependencies), Docker reuses the cached
  `pip install` layer. This saves 1-3 minutes per build.
- The `pip install` layer is only rebuilt when `requirements.txt` changes.

**`--no-cache-dir`** tells pip to not cache downloaded packages. In a Docker image, the
pip cache wastes space (it would be baked into the image layer forever). This reduces
image size by 50-200 MB.

### Lines 6-8: Copy Source Code and Configs

```dockerfile
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY params.yaml ${LAMBDA_TASK_ROOT}/
```

These are separated from the requirements install to maximize layer caching. Source code
changes frequently; dependencies change rarely.

The order is intentional:
1. `src/` -- The application code
2. `configs/` -- Configuration files (feature schemas, model parameters)
3. `params.yaml` -- Hyperparameter configuration

### Line 10: Copy Models

```dockerfile
COPY models/ ${LAMBDA_TASK_ROOT}/models/
```

This copies the trained model file (`models/model.pkl`) into the image. The model is
baked into the Docker image, which means:

**Pros:**
- No cold start delay from downloading the model from S3
- The model version is immutably tied to the image version
- Works without S3 access (simpler IAM permissions)

**Cons:**
- Image size increases (model files can be large)
- Updating the model requires building a new image
- Cannot A/B test models without separate images

### Line 12: `CMD`

```dockerfile
CMD ["src.serving.lambda_handler.handler"]
```

This tells Lambda which function to call when the Lambda function is invoked. The format
is `module.path.function_name`. Lambda imports `src.serving.lambda_handler` and calls the
`handler` function with the event and context objects.

The `CMD` uses the "exec form" (JSON array) rather than the "shell form" (plain string).
Exec form is preferred because:
- No shell process overhead
- Signals (SIGTERM) are sent directly to the process
- No shell injection vulnerabilities

### .dockerignore -- Keeping the Image Clean

```
.git
.github
.venv
venv
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.ruff_cache
mlruns
mlartifacts
data/raw
notebooks
guide
infrastructure
tests
*.md
.env
AWS_Credential
.DS_Store
```

**Every excluded pattern and why:**

| Pattern          | Why Excluded                                                    |
|------------------|-----------------------------------------------------------------|
| `.git`           | Git history can be hundreds of MB. Not needed at runtime.       |
| `.github`        | GitHub Actions workflows. Not needed at runtime.                |
| `.venv`, `venv`  | Virtual environment. Dependencies are installed fresh in Docker.|
| `__pycache__`, `*.pyc` | Python bytecode caches. Rebuilt automatically.            |
| `.pytest_cache`  | Test result caches. Not needed in production.                   |
| `.mypy_cache`    | Type checker caches. Not needed in production.                  |
| `.ruff_cache`    | Linter caches. Not needed in production.                        |
| `mlruns`, `mlartifacts` | MLflow experiment data. Not needed in production.        |
| `data/raw`       | Raw data files. Can be gigabytes. Not needed for inference.     |
| `notebooks`      | Jupyter notebooks. For exploration, not production.             |
| `guide`          | Documentation. Not needed at runtime.                           |
| `infrastructure` | Terraform files. Not needed at runtime.                         |
| `tests`          | Test files. Not needed in production.                           |
| `*.md`           | Markdown files (README, guides). Not needed at runtime.         |
| `.env`           | Environment variables file. MAY CONTAIN SECRETS. Never bake into image. |
| `AWS_Credential` | AWS credentials file. MUST be excluded for security.            |
| `.DS_Store`      | macOS metadata files. Not needed at runtime.                    |

**The `.env` and `AWS_Credential` exclusions are critical security measures.** If these
files are accidentally included in the Docker image, anyone who pulls the image can
extract your credentials.

---

## 4. Shell Scripts

### 4.1 setup_aws.sh -- Infrastructure Bootstrap

```bash
#!/bin/bash
set -euo pipefail

REGION="us-east-1"
ACCOUNT_ID="011015903780"
BUCKET="mlops-fraud-detection-${ACCOUNT_ID}"
ECR_REPO="mlops-fraud-detection"
```

#### `set -euo pipefail` -- Bash Strict Mode

This is the single most important line in any production bash script. Each flag:

**`-e` (errexit):** Exit immediately if ANY command returns a non-zero exit code. Without
this, bash continues executing after a failed command, which can cause cascading failures.

Example without `-e`:
```bash
aws s3 cp model.pkl s3://bucket/  # Fails silently (wrong bucket name)
aws lambda update-function-code ... # Succeeds but uses old model!
```

With `-e`, the script stops at the first failure.

**`-u` (nounset):** Treat unset variables as errors. Without this, `$UNDEFINED_VAR`
silently expands to an empty string.

Example without `-u`:
```bash
aws s3 rm s3://$BUCKET/ --recursive  # If BUCKET is unset, this becomes:
# aws s3 rm s3:/// --recursive       # Which could delete things you don't expect!
```

With `-u`, the script errors: `BUCKET: unbound variable`.

**`-o pipefail`:** If any command in a pipeline fails, the pipeline returns that failure.
Without this:

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
```

If `get-login-password` fails, without `pipefail`, bash only checks if `docker login`
succeeded (which it would not, but the error message would be confusing). With `pipefail`,
the pipeline fails at the correct point.

#### Idempotent Bucket Creation

```bash
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
```

**`aws s3api head-bucket`** sends a HEAD request to the bucket. If the bucket exists (and
you have access), it returns 200. If it does not exist, it returns 404. This is an
**idempotent check** -- the script can be run multiple times without creating duplicate
resources.

**The `us-east-1` region quirk:** Notice there is no `--create-bucket-configuration`
flag. For `us-east-1`, the `create-bucket` command does NOT take a location constraint.
For any other region, you would need:

```bash
aws s3api create-bucket --bucket "$BUCKET" --region "eu-west-1" \
    --create-bucket-configuration LocationConstraint=eu-west-1
```

This is a well-known AWS API inconsistency. `us-east-1` is the "default" region, so
specifying a LocationConstraint for it actually causes an error.

#### Idempotent ECR Repository Creation

```bash
if aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" 2>/dev/null; then
    echo "  Repository already exists"
else
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=true
    echo "  Repository created"
fi
```

Same pattern: check if the resource exists first, then create only if needed. The
`2>/dev/null` suppresses error messages when the resource does not exist (the error
message from `describe-repositories` is noisy and confusing in the output).

#### Creating Folder Structure

```bash
for prefix in data/raw data/processed models metrics drift-reports; do
    aws s3api put-object --bucket "${BUCKET}" --key "${prefix}/" --region "${REGION}" > /dev/null
done
```

S3 does not have real directories. But creating empty objects with trailing slashes
(`data/raw/`) creates "folder placeholders" that appear as folders in the AWS Console.
This makes the bucket structure visible and navigable for team members using the UI.

### 4.2 deploy.sh -- Build, Push, Deploy

```bash
#!/bin/bash
set -euo pipefail

REGION="us-east-1"
ACCOUNT_ID="011015903780"
ECR_REPO="mlops-fraud-detection"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
LAMBDA_FUNCTION="mlops-fraud-detection-predict"
IMAGE_TAG="${1:-latest}"
```

**`IMAGE_TAG="${1:-latest}"`** -- Uses the first command-line argument as the image tag.
If no argument is provided, defaults to `latest`. The `:-` syntax is bash's "default
value" operator.

Usage: `./scripts/deploy.sh push v1.2.3` uses tag `v1.2.3`.
       `./scripts/deploy.sh push` uses tag `latest`.

#### Push Function

```bash
push() {
    echo "=== Pushing Docker image to ECR ==="
    aws ecr get-login-password --region "${REGION}" | \
        docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

    docker build -t "${ECR_REPO}:${IMAGE_TAG}" .
    docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
    docker push "${ECR_URI}:${IMAGE_TAG}"
    echo "  Pushed: ${ECR_URI}:${IMAGE_TAG}"
}
```

**ECR Login Flow:**
1. `aws ecr get-login-password` returns a temporary authentication token
2. The token is piped (`|`) to `docker login --password-stdin`
3. Docker is now authenticated to push to ECR

**Build, Tag, Push Pattern:**
1. `docker build -t "${ECR_REPO}:${IMAGE_TAG}" .` -- Build with a local tag
2. `docker tag ... "${ECR_URI}:${IMAGE_TAG}"` -- Create an ECR-compatible tag (full URI)
3. `docker push "${ECR_URI}:${IMAGE_TAG}"` -- Upload to ECR

The two-step tag process is needed because `docker build -t` creates a local tag, but
`docker push` requires the full ECR URI to know where to push.

#### Deploy Function

```bash
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
```

**The update-then-wait pattern:**

1. `update-function-code` tells Lambda to pull the new image. Lambda acknowledges the
   request immediately but the update happens asynchronously.
2. `wait function-updated` polls Lambda's state every 5 seconds until the function
   transitions from `InProgress` to `Active`. This typically takes 10-60 seconds
   depending on image size.

Without the `wait`, the script would declare success before the deployment is actually
complete.

#### Command Dispatcher

```bash
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
```

**`case`** is bash's switch statement. The first argument determines which function to
call:
- `./scripts/deploy.sh push` -- Only build and push
- `./scripts/deploy.sh deploy` -- Only update Lambda (image must already be in ECR)
- `./scripts/deploy.sh all` -- Push then deploy (using `&&` so deploy only runs if push
  succeeds)
- Anything else -- Print usage instructions and exit with error

The `*` pattern is the default case (like `default:` in a switch statement).

---

## 5. Cost Analysis

Here is what this infrastructure costs in AWS, broken down by service.

### S3 -- Storage and Requests

| Item                     | Price                    | Our Usage                | Monthly Cost |
|--------------------------|--------------------------|--------------------------|-------------|
| Standard Storage         | $0.023/GB/month          | ~1 GB (model + data)     | $0.02       |
| Standard-IA Storage      | $0.0125/GB/month         | ~5 GB (archived models)  | $0.06       |
| PUT requests             | $0.005/1000 requests     | ~1000 requests/month     | $0.01       |
| GET requests             | $0.0004/1000 requests    | ~5000 requests/month     | $0.002      |
| **S3 Total**             |                          |                          | **~$0.09**  |

### Lambda -- Compute

| Item                     | Price                    | Our Usage                | Monthly Cost |
|--------------------------|--------------------------|--------------------------|-------------|
| Invocations              | $0.20/1M requests        | ~10,000 requests/month   | Free (1M free tier) |
| Duration (512 MB)        | $0.0000083333/second     | ~10,000 x 0.5s = 5000s  | Free (400,000 GB-s free tier) |
| **Lambda Total**         |                          |                          | **$0.00**   |

Lambda's free tier includes 1 million requests and 400,000 GB-seconds per month. A
low-traffic ML API stays well within this. At 512 MB, you get 781,250 seconds of
execution per month for free (about 9 days of continuous execution).

At higher traffic (1M requests/month at 500ms each):
- Invocations: $0.20
- Duration: 1M x 0.5s x 0.5 GB = 250,000 GB-s = $2.08
- Total: ~$2.28/month

### ECR -- Container Storage

| Item                     | Price                    | Our Usage                | Monthly Cost |
|--------------------------|--------------------------|--------------------------|-------------|
| Storage                  | $0.10/GB/month           | 10 images x 0.5 GB = 5 GB| $0.50      |
| Data transfer (out)      | $0.09/GB (same region: free)| Lambda pulls: free     | $0.00       |
| **ECR Total**            |                          |                          | **~$0.50**  |

The lifecycle policy keeping only 10 images saves significant storage. Without it,
50 deployments would mean 25 GB at $2.50/month and growing.

### CloudWatch -- Monitoring

| Item                     | Price                    | Our Usage                | Monthly Cost |
|--------------------------|--------------------------|--------------------------|-------------|
| Log ingestion            | $0.50/GB                 | ~0.5 GB/month            | $0.25       |
| Log storage              | $0.03/GB/month           | ~0.5 GB (30-day retention)| $0.015     |
| Custom metrics           | $0.30/metric/month       | 4 custom metrics         | $1.20       |
| Dashboard                | $3.00/dashboard/month    | 1 dashboard              | $3.00       |
| Alarms (standard)        | $0.10/alarm/month        | 2 alarms                 | $0.20       |
| **CloudWatch Total**     |                          |                          | **~$4.67**  |

**Note:** The dashboard is the most expensive item at $3.00/month. For cost-sensitive
projects, you could use Grafana (open source) with CloudWatch as a data source instead.
Custom metrics are also notable at $0.30 each. If you publish 20 custom metrics, that is
$6.00/month.

### API Gateway -- HTTP API

| Item                     | Price                    | Our Usage                | Monthly Cost |
|--------------------------|--------------------------|--------------------------|-------------|
| First 300M requests      | $1.00/1M requests        | ~10,000 requests         | $0.01       |
| **API Gateway Total**    |                          |                          | **~$0.01**  |

HTTP APIs are remarkably cheap. Even at 1M requests/month, the cost is only $1.00.

### Total Monthly Cost

| Service        | Cost      |
|----------------|-----------|
| S3             | $0.09     |
| Lambda         | $0.00     |
| ECR            | $0.50     |
| CloudWatch     | $4.67     |
| API Gateway    | $0.01     |
| **Total**      | **~$5.27**|

**Key insight for interviews:** This entire ML serving infrastructure costs about $5/month
for low traffic. Serverless architectures have near-zero cost at low scale and scale
linearly with usage. Compare this to a dedicated EC2 instance ($15-30/month minimum) or
an ECS Fargate cluster ($30-50/month minimum) that runs 24/7 regardless of traffic.

**At scale (1M requests/month):**

| Service        | Cost       |
|----------------|------------|
| S3             | $0.50      |
| Lambda         | $2.28      |
| ECR            | $0.50      |
| CloudWatch     | $8.00      |
| API Gateway    | $1.00      |
| **Total**      | **~$12.28**|

Still remarkably cheap. The break-even point where a dedicated EC2 instance becomes
cheaper is roughly 5-10 million requests per month (depending on instance type).

---

## Summary: Infrastructure Architecture

```
                    Internet
                       |
                       v
              +------------------+
              |  API Gateway     |
              |  (HTTP API)      |
              |  /predict (POST) |
              |  /health  (GET)  |
              +--------+---------+
                       |
                       v
              +------------------+
              |  Lambda Function |
              |  (Docker image)  |
              |  512 MB, 60s     |
              +---+----------+---+
                  |          |
                  v          v
         +--------+--+  +---+----------+
         | S3 Bucket  |  | CloudWatch   |
         | (data,     |  | (logs,       |
         |  models,   |  |  metrics,    |
         |  reports)  |  |  alarms,     |
         +------------+  |  dashboard)  |
                         +--------------+

         +------------+
         | ECR        |
         | (Docker    | <-- deploy.sh pushes images
         |  images)   |
         +------------+
```

**How it all connects:**

1. A client sends a POST request to `/predict`
2. API Gateway routes it to the Lambda function
3. Lambda runs the Docker container (pulled from ECR)
4. The container loads the model (baked into the image or from S3)
5. The container returns the prediction
6. Lambda writes logs to CloudWatch and publishes custom metrics
7. CloudWatch alarms fire if errors spike or drift is detected

**Terraform manages everything:** S3, ECR, Lambda, API Gateway, CloudWatch, IAM roles and
policies. The entire infrastructure can be created with `terraform apply` and destroyed
with `terraform destroy`.

**Interview tip:** When asked about ML infrastructure, emphasize these principles:
1. **Infrastructure as Code** -- everything is reproducible
2. **Least privilege** -- Lambda only has the permissions it needs
3. **Cost optimization** -- lifecycle rules, log retention, serverless architecture
4. **Observability** -- dashboards and alarms for both infrastructure AND model health
5. **Security** -- encryption at rest, no public access, secret management
