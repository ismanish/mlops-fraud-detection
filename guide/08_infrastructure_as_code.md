# 08 — Infrastructure as Code with Terraform

## Table of Contents
1. [Why IaC Matters for MLOps](#why-iac-matters)
2. [Terraform Basics](#terraform-basics)
3. [Our Terraform Modules Walkthrough](#our-terraform-modules)
4. [State Management](#state-management)
5. [Terraform Best Practices for ML Infrastructure](#best-practices)
6. [Cost Estimation and Optimization](#cost-estimation)
7. [Comparison: Terraform vs CloudFormation vs Pulumi vs CDK](#iac-comparison)
8. [Interview Questions](#interview-questions)

---

## Why IaC Matters for MLOps

Infrastructure as Code means defining cloud resources (servers, databases, networks,
Lambda functions) in version-controlled configuration files instead of clicking through
the AWS console.

### The Problem IaC Solves

```
Without IaC:                          With IaC:

Developer A clicks through            Developer A writes main.tf
AWS console to create Lambda           +-----------+
     |                                 | resource  |
     v                                 | "lambda"  |
Undocumented resources                 | { ... }   |
No one knows the config                +-----------+
     |                                      |
     v                                      v
Developer B needs to                   git commit + git push
replicate for staging                       |
     |                                      v
     v                                 terraform apply
"What settings did you use?"           (identical in any environment)
Manual recreation, errors                   |
     |                                      v
     v                                 Staging env matches production
Configuration drift                    exactly. Changes are reviewed.
Production != Staging                  Full audit trail in git.
```

### Why IaC Is Especially Important for MLOps

1. **Reproducibility** -- ML experiments require consistent infrastructure. If staging
   has a different Lambda memory setting than production, model latency benchmarks are
   meaningless.

2. **Environment Parity** -- Create identical dev/staging/production environments with
   a single variable change (`environment = "staging"` vs `"production"`).

3. **Disaster Recovery** -- If someone accidentally deletes the Lambda function, run
   `terraform apply` and it is recreated with the exact same configuration.

4. **Audit Trail** -- Every infrastructure change goes through code review. "Who
   changed the Lambda timeout from 30s to 60s?" is answered by `git log`.

5. **Cost Control** -- IaC makes resource creation intentional. No orphaned resources
   from manual experimentation that silently accumulate costs.

---

## Terraform Basics

### Core Concepts

```
+-------------------+       +-------------------+       +------------------+
| Configuration     | ----> | Terraform Plan    | ----> | Cloud Resources  |
| (.tf files)       |       | (preview changes) |       | (actual infra)   |
+-------------------+       +-------------------+       +------------------+
        |                           |                           |
        v                           v                           v
  Declarative:              Diff current vs              API calls to
  "I want this              desired state               AWS/GCP/Azure
   to exist"                                            to create/update/
                                                        delete resources
```

### Key Terraform Components

**Providers** -- Plugins that interface with cloud APIs:

```hcl
# infrastructure/terraform/main.tf
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

The `~> 5.0` version constraint means "any 5.x version but not 6.0". This prevents
breaking changes from provider updates while allowing patch and minor updates.

**Resources** -- The infrastructure objects you want to create:

```hcl
resource "aws_lambda_function" "prediction" {
  function_name = "${var.project_name}-predict"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.mlops.repository_url}:latest"
  timeout       = 60
  memory_size   = 512
}
```

Each resource has a type (`aws_lambda_function`) and a local name (`prediction`).
Together they form an address: `aws_lambda_function.prediction`.

**Variables** -- Parameterize your configuration:

```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}
```

**Outputs** -- Expose values after apply:

```hcl
output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.api.api_endpoint
}
```

**State** -- Terraform's record of what resources it manages (covered in detail below).

### The Terraform Workflow

```
terraform init          Download providers and modules
    |
    v
terraform plan          Preview what will change (dry run)
    |                   "Plan: 8 to add, 0 to change, 0 to destroy"
    v
terraform apply         Create/update/delete resources
    |                   "Apply complete! Resources: 8 added"
    v
terraform destroy       Tear down all managed resources
                        (use with extreme caution)
```

### Plan Output Example

```
$ terraform plan

Terraform will perform the following actions:

  # aws_lambda_function.prediction will be created
  + resource "aws_lambda_function" "prediction" {
      + function_name = "mlops-fraud-detection-predict"
      + memory_size   = 512
      + package_type  = "Image"
      + timeout       = 60
      + image_uri     = "011015903780.dkr.ecr.us-east-1.amazonaws.com/mlops-fraud-detection:latest"
    }

  # aws_apigatewayv2_api.api will be created
  + resource "aws_apigatewayv2_api" "api" {
      + name          = "mlops-fraud-detection-api"
      + protocol_type = "HTTP"
    }

Plan: 8 to add, 0 to change, 0 to destroy.
```

The `+` prefix means "will be created". `~` means "will be modified in-place".
`-` means "will be destroyed". `-/+` means "must be destroyed and recreated".

---

## Our Terraform Modules Walkthrough

Our infrastructure lives in `infrastructure/terraform/` with five files:

```
infrastructure/terraform/
+-- main.tf           # Provider configuration
+-- variables.tf      # Input variables
+-- s3.tf             # S3 bucket for data and models
+-- ecr.tf            # ECR repository for Docker images
+-- lambda.tf         # Lambda function + API Gateway + IAM
+-- cloudwatch.tf     # Log groups, dashboard, alarms
+-- outputs.tf        # Exported values
```

### 1. S3 Bucket (`s3.tf`)

Our S3 bucket stores training data, model artifacts, and drift reports.

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

**Bucket naming:** `mlops-fraud-detection-011015903780`. Including the account ID
ensures global uniqueness (S3 bucket names are globally unique across all AWS
accounts).

**Versioning:**

```hcl
resource "aws_s3_bucket_versioning" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  versioning_configuration {
    status = "Enabled"
  }
}
```

Every object overwrite creates a new version. This is critical for ML: if a new model
is deployed and performs worse, you can roll back to any previous model version.

**Encryption:**

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

All objects are encrypted at rest using AES-256 (SSE-S3). For more sensitive data,
you would use SSE-KMS with a customer-managed key.

**Public Access Block:**

```hcl
resource "aws_s3_bucket_public_access_block" "mlops" {
  bucket                  = aws_s3_bucket.mlops.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

All four public access settings are blocked. ML data and models should never be
publicly accessible.

**Lifecycle Rules:**

```hcl
resource "aws_s3_bucket_lifecycle_configuration" "mlops" {
  bucket = aws_s3_bucket.mlops.id

  rule {
    id     = "archive-old-models"
    status = "Enabled"
    filter { prefix = "models/" }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "expire-old-drift-reports"
    status = "Enabled"
    filter { prefix = "drift-reports/" }
    expiration { days = 365 }
  }
}
```

Two lifecycle rules for cost optimization:
- Models older than 90 days move to Infrequent Access (cheaper storage, same
  durability). We keep them for rollback capability but rarely access them.
- Drift reports older than 365 days are deleted. They are only useful for
  historical analysis, which typically looks at the last few months.

### 2. ECR Repository (`ecr.tf`)

```hcl
resource "aws_ecr_repository" "mlops" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
```

**`image_tag_mutability = "MUTABLE"`** allows overwriting the `latest` tag. This is
convenient for development but risky for production. In a mature setup, you would use
`"IMMUTABLE"` and deploy only with commit-SHA tags.

**`scan_on_push = true`** runs Amazon Inspector vulnerability scanning on every
pushed image. This catches known CVEs in OS packages and Python dependencies.

**Lifecycle Policy:**

```hcl
resource "aws_ecr_lifecycle_policy" "mlops" {
  repository = aws_ecr_repository.mlops.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
```

Keeps only the 10 most recent images. Each image is ~350MB compressed, so 10 images
use ~3.5GB of ECR storage ($0.10/GB/month = $0.35/month). Without this policy, images
would accumulate indefinitely.

### 3. Lambda + API Gateway + IAM (`lambda.tf`)

This is the largest and most complex Terraform file, defining the complete serving
infrastructure.

**IAM Role:**

```
+---------------------+       +------------------------+
| Lambda Function     | ----> | IAM Role               |
| (needs permissions  |       | - AssumeRole (Lambda)  |
|  to run)            |       | - S3: Get/Put/List     |
|                     |       | - CloudWatch: PutMetric|
|                     |       | - Basic Execution      |
+---------------------+       +------------------------+
```

The role follows least-privilege principle:
- `AWSLambdaBasicExecutionRole` -- Write logs to CloudWatch Logs
- S3 access limited to our specific bucket (not `*`)
- CloudWatch PutMetricData for custom metrics

**Lambda Function:**

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

  depends_on = [aws_ecr_repository.mlops]
}
```

Key configuration choices:
- **`timeout = 60`** -- 60 seconds. Cold starts can take 5-8 seconds, and inference
  takes ~100ms. 60 seconds provides ample headroom.
- **`memory_size = 512`** -- 512 MB. XGBoost with our model size needs ~200-300MB.
  512MB gives headroom and allocates proportional CPU (Lambda allocates CPU linearly
  with memory; 512MB gets ~0.3 vCPU).
- **`depends_on`** -- Explicit dependency on the ECR repository. Terraform must create
  the repository before the Lambda function can reference its image.

**API Gateway:**

The API Gateway configuration defines the HTTP API, stage, integration, and routes.
The flow of a request through these resources:

```
Internet --> API Gateway API --> Stage ($default) --> Route (POST /predict)
                                                          |
                                                          v
                                                  Integration (AWS_PROXY)
                                                          |
                                                          v
                                                  Lambda Function
```

**Lambda Permission:**

```hcl
resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.prediction.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
```

This grants API Gateway permission to invoke our Lambda function. Without this,
API Gateway would receive a 403 error from Lambda. The `/*/*` wildcard allows any
HTTP method on any route from this API.

### 4. CloudWatch (`cloudwatch.tf`)

Two log groups, one dashboard, two alarms. See the Monitoring guide (07) for
detailed explanation of the dashboard widgets and alarm configurations.

**Log retention:**

```hcl
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project_name}-predict"
  retention_in_days = 30
}
```

30-day retention balances cost with debugging needs. Lambda logs can grow large
under high traffic. Without a retention policy, logs are kept forever.

### 5. Variables (`variables.tf`)

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

All variables have defaults, making `terraform apply` work without a `.tfvars` file.
For multi-environment deployment, you would override with a file:

```bash
# staging.tfvars
environment = "staging"
project_name = "mlops-fraud-detection-staging"
```

```bash
terraform apply -var-file="staging.tfvars"
```

### 6. Outputs (`outputs.tf`)

```hcl
output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "cloudwatch_dashboard_url" {
  value = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${var.project_name}-dashboard"
}
```

Five outputs expose the most-needed values after deployment. These can be consumed
by other Terraform modules, scripts, or displayed to the operator after `terraform
apply`.

### Resource Dependency Graph

```
                    +------------------+
                    |   variables.tf   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
        +---------+   +-----------+   +-----------+
        | s3.tf   |   | ecr.tf    |   |cloudwatch |
        | (bucket)|   | (repo)    |   | .tf (logs)|
        +----+----+   +-----+-----+   +-----+-----+
             |              |              |
             +--------------+--------------+
                            |
                            v
                     +------+------+
                     | lambda.tf   |
                     | (function,  |
                     |  API GW,    |
                     |  IAM role)  |
                     +------+------+
                            |
                            v
                     +------+------+
                     | outputs.tf  |
                     +-------------+
```

---

## State Management

### What Is Terraform State?

Terraform state (`terraform.tfstate`) is a JSON file that maps your configuration
to real cloud resources. It stores resource IDs, attributes, and metadata.

```json
{
  "resources": [
    {
      "type": "aws_lambda_function",
      "name": "prediction",
      "instances": [
        {
          "attributes": {
            "function_name": "mlops-fraud-detection-predict",
            "arn": "arn:aws:lambda:us-east-1:011015903780:function:mlops-fraud-detection-predict",
            "memory_size": 512,
            "timeout": 60
          }
        }
      ]
    }
  ]
}
```

### Why State Matters

Without state, Terraform cannot:
- Know what resources it already created (it would try to create duplicates)
- Detect drift between configuration and reality
- Determine the correct order for resource destruction

### Local vs Remote State

**Local State (our current setup):**

```
Developer Machine
+-----------------------------------+
| infrastructure/terraform/         |
|   main.tf                         |
|   terraform.tfstate  <-- HERE     |
|   terraform.tfstate.backup        |
+-----------------------------------+
```

Problems:
- State file is on one person's machine
- No locking -- two people running `terraform apply` simultaneously corrupt state
- State contains sensitive data (ARNs, IDs) that should not be in git

**Remote State (production recommendation):**

```hcl
terraform {
  backend "s3" {
    bucket         = "mlops-terraform-state"
    key            = "fraud-detection/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

```
                    +-------------------+
                    |    S3 Bucket      |
                    | (terraform state) |
                    +--------+----------+
                             |
               +-------------+-------------+
               |                           |
        +------v------+            +-------v------+
        | Developer A |            | Developer B  |
        | terraform   |            | terraform    |
        | apply       |            | apply        |
        +------+------+            +-------+------+
               |                           |
               +-------------+-------------+
                             |
                    +--------v----------+
                    |  DynamoDB Table    |
                    |  (state locking)  |
                    +-------------------+
```

Benefits:
- **Shared access** -- All team members read the same state
- **Locking** -- DynamoDB prevents concurrent modifications
- **Encryption** -- State is encrypted at rest in S3
- **Versioning** -- S3 versioning enables state rollback

### State Commands

```bash
# List all resources in state
terraform state list

# Show details of a specific resource
terraform state show aws_lambda_function.prediction

# Remove a resource from state (without destroying it)
terraform state rm aws_lambda_function.prediction

# Import an existing resource into state
terraform import aws_s3_bucket.mlops mlops-fraud-detection-011015903780

# Move a resource in state (after renaming in config)
terraform state mv aws_lambda_function.old aws_lambda_function.new
```

---

## Terraform Best Practices for ML Infrastructure

### 1. Tag Everything

```hcl
tags = {
  Project     = var.project_name
  Environment = var.environment
  ManagedBy   = "terraform"
}
```

Our project tags every resource with three tags. `ManagedBy = "terraform"` tells
anyone viewing the AWS console that this resource is managed by IaC and should not
be modified manually.

### 2. Use Variables for Anything That Changes Between Environments

```
Production:  memory_size = 512,  timeout = 60
Staging:     memory_size = 256,  timeout = 30
Development: memory_size = 128,  timeout = 15
```

### 3. Pin Provider Versions

```hcl
required_providers {
  aws = {
    source  = "hashicorp/aws"
    version = "~> 5.0"    # Allows 5.x, blocks 6.0
  }
}
```

Unpinned providers can introduce breaking changes. Use `~>` for minor version
flexibility or `=` for exact pinning.

### 4. Use Separate State Files Per Environment

```
terraform/
+-- environments/
    +-- production/
    |   +-- main.tf
    |   +-- backend.tf     (state in s3://tf-state/prod/)
    +-- staging/
        +-- main.tf
        +-- backend.tf     (state in s3://tf-state/staging/)
```

This prevents a staging change from accidentally affecting production.

### 5. Use `depends_on` Sparingly

Terraform infers most dependencies from resource references. Explicit `depends_on`
is only needed when there is an implicit dependency:

```hcl
# Terraform knows Lambda depends on IAM role because we reference role.arn
resource "aws_lambda_function" "prediction" {
  role = aws_iam_role.lambda_role.arn  # Implicit dependency
}

# But ECR repo must exist before Lambda can pull from it
# The image_uri reference does NOT create a dependency because
# the repo URL exists before any image is pushed
resource "aws_lambda_function" "prediction" {
  depends_on = [aws_ecr_repository.mlops]  # Explicit dependency
}
```

### 6. Use Lifecycle Rules for ML Resources

```hcl
resource "aws_lambda_function" "prediction" {
  # ...

  lifecycle {
    ignore_changes = [image_uri]  # CI/CD updates this outside Terraform
  }
}
```

The `ignore_changes` lifecycle rule is important for ML: the CI/CD pipeline updates
the Lambda image URI on every deployment, but Terraform should not revert it to
`latest` on the next `terraform apply`.

### 7. Validate Before Apply

```bash
terraform fmt -check        # Check formatting
terraform validate          # Check syntax and type errors
terraform plan -out=plan    # Save plan to file
terraform apply plan        # Apply exact plan (no prompt)
```

In CI/CD, always run `plan` on pull requests and `apply` only on merge to main.

---

## Cost Estimation and Optimization

### Our Monthly Cost Estimate

```
+----------------------------+-----------+---------------------------+
| Resource                   | Monthly $ | Notes                     |
+----------------------------+-----------+---------------------------+
| Lambda (100K req/day)      | $0.00     | Within free tier          |
| API Gateway (3M req/month) | $3.00     | $1/million requests       |
| ECR (3.5 GB storage)       | $0.35     | $0.10/GB                  |
| S3 (10 GB, Standard)       | $0.23     | $0.023/GB                 |
| CloudWatch Logs (5 GB)     | $2.50     | $0.50/GB ingestion        |
| CloudWatch Metrics (6)     | $1.80     | $0.30/metric              |
| CloudWatch Dashboard (1)   | $3.00     | $3/dashboard              |
| CloudWatch Alarms (2)      | $0.20     | $0.10/alarm               |
+----------------------------+-----------+---------------------------+
| TOTAL                      | ~$11/month|                           |
+----------------------------+-----------+---------------------------+
```

### Cost Optimization Strategies

**1. S3 Lifecycle Rules** (already implemented)
- Move old models to IA: saves ~40% on storage
- Delete old drift reports: prevents unbounded growth

**2. ECR Lifecycle Policy** (already implemented)
- Keep only 10 images: prevents ~$1/month growth per deployment

**3. CloudWatch Log Retention** (already implemented)
- 30-day retention prevents log storage from growing indefinitely
- Without retention: 5 GB/month x 12 months = 60 GB = $30/year in log storage

**4. Lambda Right-Sizing**
- Monitor actual memory usage via CloudWatch `Max Memory Used`
- If the function consistently uses 200MB, reduce from 512MB to 256MB
- This also reduces cost since Lambda charges per GB-second

**5. Reserved Concurrency (cost ceiling)**

```hcl
resource "aws_lambda_function" "prediction" {
  reserved_concurrent_executions = 50  # Cap at 50 concurrent
}
```

This prevents runaway costs from DDoS or misconfigured clients.

### Terraform Cost Estimation Tools

```bash
# Infracost -- estimate costs before deploying
infracost breakdown --path=infrastructure/terraform/

# Output:
# NAME                                    MONTHLY COST
# aws_lambda_function.prediction          $0.00
# aws_apigatewayv2_api.api                $3.00
# aws_cloudwatch_dashboard.mlops          $3.00
# ...
# TOTAL                                   $11.08
```

---

## Comparison: Terraform vs CloudFormation vs Pulumi vs CDK

```
+-------------------+---------------+----------------+-------------+-------------+
| Feature           | Terraform     | CloudFormation | Pulumi      | CDK         |
+-------------------+---------------+----------------+-------------+-------------+
| Language          | HCL           | JSON/YAML      | Python/TS/  | Python/TS/  |
|                   |               |                | Go/C#       | Java/Go/C#  |
+-------------------+---------------+----------------+-------------+-------------+
| Multi-Cloud       | Yes           | No (AWS only)  | Yes         | No (AWS)    |
+-------------------+---------------+----------------+-------------+-------------+
| State Management  | Self-managed  | AWS-managed    | Self/Cloud  | AWS-managed |
|                   | (S3 backend)  | (CloudFormation|             | (via CFN)   |
|                   |               |  service)      |             |             |
+-------------------+---------------+----------------+-------------+-------------+
| Learning Curve    | Medium (HCL)  | Low (YAML)     | Low (code)  | Medium      |
+-------------------+---------------+----------------+-------------+-------------+
| Community/Modules | Largest       | Moderate       | Growing     | Growing     |
+-------------------+---------------+----------------+-------------+-------------+
| Drift Detection   | terraform plan| Drift detection| pulumi      | CFN drift   |
|                   | (manual)      | (built-in)     | preview     | detection   |
+-------------------+---------------+----------------+-------------+-------------+
| Preview Changes   | terraform plan| Change sets    | pulumi      | cdk diff    |
|                   |               |                | preview     |             |
+-------------------+---------------+----------------+-------------+-------------+
| Rollback          | Manual (apply | Automatic      | Manual      | Automatic   |
|                   |  previous     | (stack         |             | (via CFN)   |
|                   |  state)       |  rollback)     |             |             |
+-------------------+---------------+----------------+-------------+-------------+
| Cost              | Free (OSS)    | Free           | Free/Paid   | Free        |
|                   | Paid (Cloud)  |                | (team feat.)|             |
+-------------------+---------------+----------------+-------------+-------------+
```

### Why We Chose Terraform

1. **Multi-cloud portability** -- If we move from AWS to GCP, we change the provider,
   not the tool. HCL concepts transfer across clouds.
2. **Largest ecosystem** -- The Terraform Registry has thousands of community modules
   for common patterns.
3. **Industry standard** -- Most MLOps job postings list Terraform. It is the most
   widely adopted IaC tool.
4. **Declarative HCL** -- Easier to read and review than imperative code (Pulumi) or
   verbose YAML (CloudFormation). A `terraform plan` diff is intuitive for code review.
5. **State management flexibility** -- Choose local state for learning, S3 backend for
   teams, or Terraform Cloud for enterprise features.

### When to Choose Alternatives

| Scenario | Best Choice |
|----------|------------|
| AWS-only shop, team knows Python | CDK |
| AWS-only, prefer YAML, want automatic rollback | CloudFormation |
| Multi-cloud, need real programming (loops, conditions) | Pulumi |
| Multi-cloud, industry standard, largest community | Terraform |
| Kubernetes-focused infrastructure | Pulumi or Terraform with Helm provider |

---

## Interview Questions

### Q1: What is Terraform state and why is it important?

**A:** Terraform state is a JSON file that maps your HCL configuration to real cloud
resources. It stores the resource IDs, attributes, and dependency graph so Terraform
can determine what needs to be created, updated, or destroyed on the next `apply`.
Without state, Terraform would not know that `aws_lambda_function.prediction` in your
config corresponds to the actual Lambda function
`arn:aws:lambda:us-east-1:011015903780:function:mlops-fraud-detection-predict` in AWS.
State also enables Terraform to detect drift -- if someone manually changes the Lambda
memory in the console, `terraform plan` will show the discrepancy and propose to
correct it. In our project, we would use S3 as a remote backend with DynamoDB locking
for team collaboration.

### Q2: What happens if two team members run `terraform apply` simultaneously?

**A:** With local state, this is a race condition that can corrupt state -- one apply
may overwrite the other's changes, leading to orphaned resources or missing resources
in state. The solution is remote state with locking. Using our S3 backend with a
DynamoDB table for locking, the first `terraform apply` acquires a lock. The second
developer's `apply` immediately fails with "Error acquiring the state lock" and a
message showing who holds the lock. The lock is released when the first apply
completes. This is why DynamoDB is paired with S3 for state management -- S3 provides
durable storage, and DynamoDB provides atomic locking.

### Q3: Explain the difference between `terraform plan` and `terraform apply`.

**A:** `terraform plan` is a read-only operation that compares the desired state
(your .tf files) against the current state (terraform.tfstate) and the actual cloud
resources (via API calls). It outputs a human-readable diff showing what will be
created (+), modified (~), or destroyed (-). No changes are made. `terraform apply`
executes the plan, making actual API calls to create, update, or delete resources.
Best practice is to save the plan to a file (`terraform plan -out=plan.tfplan`) and
then apply that exact plan (`terraform apply plan.tfplan`), ensuring that what you
reviewed is exactly what gets applied. In CI/CD, we would run `plan` on pull requests
for review and `apply` only after merge to main.

### Q4: How do you handle secrets in Terraform?

**A:** Never hardcode secrets in .tf files or commit them to git. We use several
approaches: (1) Environment variables like `AWS_ACCESS_KEY_ID` that Terraform reads
automatically, (2) AWS Secrets Manager or SSM Parameter Store for application secrets,
referenced in Terraform with `data "aws_ssm_parameter"` blocks, (3) `.tfvars` files
listed in `.gitignore` for sensitive variable values, (4) Terraform Cloud or Vault
for team secret management. In our project, the AWS account ID is in `variables.tf`
(not a secret), but AWS credentials are provided via environment variables in GitHub
Actions using repository secrets.

### Q5: Why do you use `lifecycle { ignore_changes = [image_uri] }` for Lambda?

**A:** Our CI/CD pipeline (`deploy.yml`) updates the Lambda function's image URI on
every deployment via `aws lambda update-function-code`. This happens outside
Terraform's control. Without `ignore_changes`, the next `terraform apply` would see
the image_uri has changed from `latest` (in the config) to the commit-SHA-tagged
image (set by CI/CD) and revert it back to `latest`, undoing the deployment. The
`ignore_changes` lifecycle rule tells Terraform to skip this attribute during
planning, letting CI/CD own the image version while Terraform owns the infrastructure
configuration. This is a common pattern for ML deployments where the model artifact
changes more frequently than the infrastructure.

### Q6: How would you set up multi-environment infrastructure?

**A:** I would use workspaces or separate state files for each environment. The
preferred approach is separate directories with shared modules:

```
modules/
  serving/           # Reusable module with Lambda, API GW, etc.
    main.tf
    variables.tf

environments/
  production/
    main.tf          # module "serving" { source = "../../modules/serving" }
    terraform.tfvars # environment = "production", memory = 512
  staging/
    main.tf
    terraform.tfvars # environment = "staging", memory = 256
```

Each environment has its own state file and can be applied independently. Shared
modules ensure consistency while variables allow environment-specific tuning (e.g.,
smaller Lambda memory in staging to save costs). I prefer this over Terraform
workspaces because workspaces share the same backend configuration, making it
easier to accidentally apply staging changes to production.

### Q7: What is the purpose of the `depends_on` in our Lambda resource?

**A:** Our Lambda function has `depends_on = [aws_ecr_repository.mlops]` because
Terraform needs to create the ECR repository before Lambda can reference its image
URI. Normally, Terraform infers dependencies from resource attribute references --
for example, `role = aws_iam_role.lambda_role.arn` automatically creates a dependency
on the IAM role. But the ECR dependency is special: the `image_uri` references the
repository URL, which Terraform knows at plan time (it is deterministic from the
repository name and account), so Terraform does not automatically infer the ordering.
The explicit `depends_on` ensures the repository exists and is ready to serve images
before the Lambda function is created.

### Q8: How does Terraform handle resource destruction order?

**A:** Terraform builds a dependency graph and destroys resources in reverse
dependency order. In our project, `terraform destroy` would proceed roughly as:
(1) Remove Lambda permission (API GW -> Lambda link), (2) Remove API Gateway routes
and integrations, (3) Remove API Gateway, (4) Remove Lambda function, (5) Remove
IAM role and policies, (6) Remove ECR repository, (7) Remove S3 bucket, (8) Remove
CloudWatch resources. It cannot delete the IAM role before the Lambda function
because the Lambda still references it. If destruction fails mid-way (e.g., S3
bucket is not empty), Terraform records the partial state and can be re-run after
fixing the issue (e.g., emptying the bucket manually).

### Q9: Compare Terraform with AWS CDK for ML infrastructure.

**A:** Terraform uses the declarative HCL language, which is cloud-agnostic and
produces readable infrastructure definitions that anyone can review. CDK uses
imperative programming languages (Python, TypeScript) that compile down to
CloudFormation templates, providing full programming power (loops, conditionals,
inheritance) but only for AWS. For ML infrastructure specifically, CDK's Python
support feels natural to data scientists, and its higher-level constructs (like
`LambdaRestApi` which creates API Gateway + Lambda + permissions in one line) reduce
boilerplate. However, Terraform's multi-cloud support is valuable if you use GCP for
training (Vertex AI) and AWS for serving, and its plan/apply workflow is more
transparent than CDK's synthesize/deploy process. I chose Terraform for this project
because it is the industry standard for MLOps roles and the skill transfers across
cloud providers.

### Q10: How would you implement blue-green deployment with Terraform?

**A:** I would create two Lambda function resources (blue and green), each pointing
to different container images. API Gateway would route to the active one via a
weighted integration. To deploy: (1) Update the inactive function's image URI,
(2) Run a canary test against the inactive function directly (via invoke API),
(3) Flip the API Gateway route to the new function, (4) Monitor for errors, (5) If
issues arise, flip the route back (instant rollback). In Terraform, this means
having a variable like `active_color = "blue"` that controls which Lambda the API
Gateway route targets. Changing the variable and running `terraform apply` performs
the switch. Lambda aliases with weighted routing are an even simpler approach --
shift traffic gradually from 0% to 100% on the new version.
