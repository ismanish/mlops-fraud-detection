# Containerization for ML with Docker

## Why Containers Matter for ML

Machine learning has a notorious "dependency hell" problem. A typical ML project depends on:

- A specific Python version (3.10 vs 3.12 can break things)
- NumPy, pandas, scikit-learn (version mismatches cause silent numerical differences)
- XGBoost (C++ backend, compilation flags matter)
- System-level libraries (libblas, libgomp, libstdc++)
- CUDA/cuDNN for GPU models (exact version matching required)

The classic scenario: a model trained on a data scientist's MacBook with Python 3.11,
XGBoost 2.0.3, and NumPy 1.24 produces AUC-ROC 0.97. The same model deployed to a Linux
server with Python 3.12, XGBoost 2.1.0, and NumPy 1.26 produces AUC-ROC 0.93. Same code,
same data, different results. Different numerical libraries can use different floating-point
optimizations that produce subtly different results.

Docker solves this by packaging the entire execution environment:

```
Without Docker:
  Data Scientist's Mac     Production Server
  +------------------+    +------------------+
  | macOS 14         |    | Ubuntu 22.04     |
  | Python 3.11      |    | Python 3.12      |
  | XGBoost 2.0.3    |    | XGBoost 2.1.0    |
  | NumPy 1.24       |    | NumPy 1.26       |
  | libstdc++ 13     |    | libstdc++ 12     |
  +------------------+    +------------------+
  Result: AUC 0.97         Result: AUC 0.93
              DIFFERENT RESULTS!

With Docker:
  Data Scientist's Mac     Production Server
  +------------------+    +------------------+
  | Docker Engine    |    | Docker Engine    |
  | +==============+ |    | +==============+ |
  | | Container    | |    | | Same         | |
  | | Python 3.12  | |    | | Container    | |
  | | XGBoost 2.0  | |    | | Python 3.12  | |
  | | NumPy 1.24   | |    | | XGBoost 2.0  | |
  | | Ubuntu 22.04 | |    | | NumPy 1.24   | |
  | +==============+ |    | +==============+ |
  +------------------+    +------------------+
  Result: AUC 0.97         Result: AUC 0.97
              IDENTICAL RESULTS!
```

---

## Our Dockerfile Explained Line by Line

**File:** `Dockerfile`

```dockerfile
FROM public.ecr.aws/lambda/python:3.12
```

**Line 1: Base image.** We use the official AWS Lambda Python 3.12 runtime image from the
Amazon ECR Public Gallery. This is NOT a generic Python image --- it includes the Lambda
Runtime Interface Client (RIC) and Runtime Interface Emulator (RIE), which allow the
container to receive and respond to Lambda invocation events.

Why this base image?
- It is the only image that works with AWS Lambda container deployment
- It includes the Lambda runtime API that Mangum (our ASGI adapter) interfaces with
- It is based on Amazon Linux 2023, which is what Lambda uses natively
- It is maintained and security-patched by AWS
- It is ~300 MB, much smaller than a generic Python + Lambda combo

```dockerfile
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt
```

**Lines 3-4: Install dependencies.** `${LAMBDA_TASK_ROOT}` is an environment variable set
by the Lambda base image (defaults to `/var/task`). We copy requirements.txt first and
install before copying the rest of the code. This is a Docker layer caching optimization:

```
Layer caching strategy:

  Layer 1: FROM public.ecr.aws/lambda/python:3.12    (cached, ~300 MB)
  Layer 2: COPY requirements.txt + RUN pip install    (cached if requirements.txt unchanged)
  Layer 3: COPY src/ + configs/ + params.yaml         (rebuilt on code change)
  Layer 4: COPY models/                                (rebuilt on model change)

  If you only change code, layers 1-2 are cached (saving ~2 minutes of pip install).
  If you only change the model, layers 1-3 are cached.
```

The `--no-cache-dir` flag tells pip not to save downloaded packages. This reduces the
image size by avoiding duplicate storage (the installed packages are already in
site-packages).

```dockerfile
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY params.yaml ${LAMBDA_TASK_ROOT}/
```

**Lines 6-8: Copy application code.** We copy three things:
- `src/` --- All Python modules (data, models, serving, monitoring, utils)
- `configs/` --- Configuration files
- `params.yaml` --- The central configuration file

Note: we do NOT copy tests, notebooks, infrastructure, guide, or any other non-runtime
files. The `.dockerignore` ensures these are excluded even if someone accidentally uses
a broader COPY command.

```dockerfile
COPY models/ ${LAMBDA_TASK_ROOT}/models/
```

**Line 10: Copy the trained model.** The `models/model.pkl` file is baked into the
container image. This is a deliberate design choice:

Pros:
- The model and code are shipped as a single atomic unit
- No runtime dependency on S3 or any external storage
- Cold starts are faster (no model download on first invocation)
- Immutable: the model in image `abc123` is always the same model

Cons:
- Changing the model requires rebuilding and redeploying the container
- Image size increases by the model size (~50-100 MB for XGBoost)

For our fraud detection use case, the pros outweigh the cons. The model changes
infrequently (retraining happens weekly or on drift detection), and cold start latency
matters for real-time fraud detection.

```dockerfile
CMD ["src.serving.lambda_handler.handler"]
```

**Line 12: Entry point.** This tells the Lambda runtime to invoke the `handler` function
from `src.serving.lambda_handler`. The handler is a Mangum instance that wraps our FastAPI
app:

```python
# src/serving/lambda_handler.py
from mangum import Mangum
from src.serving.app import app
handler = Mangum(app, lifespan="on")
```

The chain is: Lambda Invoke -> Container -> Mangum -> FastAPI -> predict()

---

## Understanding the Lambda Container Architecture

```
API Gateway (HTTP Request)
    |
    v
AWS Lambda Service
    |
    v
+------------------------------------------+
| Container (our Docker image)             |
|                                          |
| Lambda Runtime Interface Client (RIC)    |
|     |                                    |
|     v                                    |
| lambda_handler.handler (Mangum)          |
|     |                                    |
|     v                                    |
| FastAPI app (src/serving/app.py)         |
|     |                                    |
|     v                                    |
| Route: POST /predict                     |
|     |                                    |
|     v                                    |
| predict(features) -> {prediction, prob}  |
|     |                                    |
|     v                                    |
| CloudWatch (publish metrics)             |
|                                          |
| Files in container:                      |
|   /var/task/src/                         |
|   /var/task/models/model.pkl             |
|   /var/task/params.yaml                  |
+------------------------------------------+
```

**Mangum** is the bridge between AWS Lambda and ASGI (FastAPI). When Lambda invokes the
handler, Mangum:
1. Receives the Lambda event (API Gateway HTTP request)
2. Converts it to an ASGI scope (standard Python web framework format)
3. Passes it to FastAPI for routing and handling
4. Converts the FastAPI response back to a Lambda response
5. Returns it to API Gateway

This allows us to use the same FastAPI app for local development (`uvicorn`) and
production (Lambda), just with a different entry point.

---

## The .dockerignore File

**File:** `.dockerignore`

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

Every entry serves a purpose:

| Entry          | Why excluded                                                    |
|----------------|-----------------------------------------------------------------|
| `.git`         | Git history is huge and irrelevant at runtime                   |
| `.github`      | CI/CD workflows not needed in container                         |
| `.venv/venv`   | Local virtual env would conflict with container's Python        |
| `__pycache__`  | Compiled bytecode from wrong Python version                     |
| `mlruns`       | MLflow experiment data (potentially gigabytes)                  |
| `data/raw`     | Raw training data not needed for inference                      |
| `notebooks`    | Jupyter notebooks are for development only                      |
| `guide`        | Documentation not needed at runtime                             |
| `infrastructure` | Terraform files not needed in container                       |
| `tests`        | Test code not needed at runtime                                 |
| `*.md`         | Documentation files                                             |
| `.env`         | Local environment variables (may contain secrets)               |
| `AWS_Credential` | AWS credentials file (must NEVER be in the image)            |

The `AWS_Credential` exclusion is critical. If this file were included in the Docker image
and pushed to ECR, anyone with access to the image could extract the credentials. Lambda
gets its AWS permissions from the IAM role (`lambda.tf`), not from credential files.

---

## Multi-Stage Builds for ML

Our current Dockerfile uses a single stage because the Lambda base image is already
optimized. However, for non-Lambda deployments, multi-stage builds can significantly
reduce image size:

```dockerfile
# Stage 1: Builder (install dependencies)
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

# Stage 2: Runtime (minimal image)
FROM python:3.12-slim AS runtime

# Copy only installed packages, not pip/setuptools/wheel
COPY --from=builder /app/dependencies /usr/local/lib/python3.12/site-packages/

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY params.yaml .
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage helps:**

```
Single-stage image:
  python:3.12       (~1.0 GB)
  + pip, setuptools (~50 MB)
  + build tools     (~200 MB)    <-- gcc, make, etc. for compiling C extensions
  + dependencies    (~500 MB)
  + our code        (~10 MB)
  = ~1.7 GB

Multi-stage image:
  python:3.12-slim  (~150 MB)
  + dependencies    (~500 MB)    <-- copied from builder, no build tools
  + our code        (~10 MB)
  = ~660 MB

Lambda base image (what we use):
  lambda/python:3.12 (~300 MB)
  + dependencies    (~500 MB)
  + our code        (~10 MB)
  = ~810 MB
```

The multi-stage build eliminates build tools (gcc, make, development headers) that are
needed to compile Python C extensions (like NumPy, XGBoost) but are not needed at runtime.

### When to use multi-stage builds for ML

- **Use for:** Non-Lambda deployments (ECS, EKS, plain Docker), models with large
  compilation dependencies (PyTorch, TensorFlow), images that will be pulled frequently
  (smaller = faster pulls)
- **Skip for:** Lambda deployments (base image is already optimized), quick prototypes,
  teams comfortable with larger images

---

## Docker + ECR Workflow

Our deployment pipeline uses ECR (Elastic Container Registry) as the Docker image registry:

```
Developer Machine / CI Runner
+-------------------------------------------+
|                                           |
| 1. docker build -t fraud-detection .      |
| 2. docker tag ... <ECR_URI>:<tag>         |
| 3. aws ecr get-login-password | \        |
|    docker login --username AWS ...         |
| 4. docker push <ECR_URI>:<tag>            |
|                                           |
+-------------------------------------------+
         |
         | (push over HTTPS)
         v
+-------------------------------------------+
| ECR Repository: mlops-fraud-detection     |
|                                           |
| Images:                                   |
|   abc123def4 (2024-01-15) 810 MB  latest  |
|   def456ghi7 (2024-01-10) 808 MB          |
|   ghi789jkl0 (2024-01-05) 805 MB          |
|   ...                                     |
|                                           |
| Lifecycle policy: keep last 10 images     |
| Scan on push: enabled (vulnerability scan)|
+-------------------------------------------+
         |
         | (Lambda pulls image)
         v
+-------------------------------------------+
| Lambda Function: mlops-fraud-prediction   |
|                                           |
| Image URI: <ECR_URI>:abc123def4           |
| Memory: 512 MB                            |
| Timeout: 60 seconds                       |
+-------------------------------------------+
```

### ECR setup from our Terraform

```hcl
# infrastructure/terraform/ecr.tf
resource "aws_ecr_repository" "mlops" {
  name                 = var.project_name    # "mlops-fraud-detection"
  image_tag_mutability = "MUTABLE"           # Allow "latest" tag to be updated

  image_scanning_configuration {
    scan_on_push = true                      # Scan for vulnerabilities on every push
  }
}

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

**Key ECR configuration decisions:**

1. **`image_tag_mutability = "MUTABLE"`** allows the `latest` tag to be reassigned to new
   images. This is convenient for development but means `latest` is not a reliable reference.
   For immutable deployments, we use the Git SHA tag.

2. **`scan_on_push = true`** automatically scans every image for known vulnerabilities
   (CVEs) when it is pushed. This catches security issues in our Python dependencies
   before they reach production.

3. **Lifecycle policy: keep last 10 images.** ECR storage costs money. Without this policy,
   images accumulate indefinitely. Keeping 10 allows rollback to recent versions while
   controlling costs. Since each image is ~800 MB, 10 images = ~8 GB of storage.

### The deploy.sh script

Our `scripts/deploy.sh` encapsulates the ECR push and Lambda deployment:

```bash
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

push() {
    # Authenticate Docker with ECR
    aws ecr get-login-password --region "${REGION}" | \
        docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

    # Build and tag
    docker build -t "${ECR_REPO}:${IMAGE_TAG}" .
    docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

    # Push to ECR
    docker push "${ECR_URI}:${IMAGE_TAG}"
}

deploy() {
    # Update Lambda to use the new image
    aws lambda update-function-code \
        --function-name "${LAMBDA_FUNCTION}" \
        --image-uri "${ECR_URI}:${IMAGE_TAG}" \
        --region "${REGION}"

    # Wait for Lambda to finish updating
    aws lambda wait function-updated \
        --function-name "${LAMBDA_FUNCTION}" \
        --region "${REGION}"
}
```

---

## Container Optimization Tips

### 1. Order COPY commands by change frequency

```dockerfile
# GOOD: least-changed files first
COPY requirements.txt .              # Changes rarely
RUN pip install -r requirements.txt  # Cached most of the time
COPY src/ ./src/                     # Changes sometimes
COPY models/ ./models/               # Changes on retrain

# BAD: everything at once
COPY . .                             # Any change invalidates ALL layers
RUN pip install -r requirements.txt  # Re-installs everything, every time
```

### 2. Pin dependency versions

```
# requirements.txt
scikit-learn>=1.3.0      # We use >= for flexibility during development
xgboost>=2.0.0

# For production, consider pinning exact versions:
scikit-learn==1.3.2
xgboost==2.0.3
```

Exact pinning ensures byte-identical builds. Range specifiers (>=) allow pip to install
different versions on different build days, potentially causing subtle differences. For
maximum reproducibility, use `pip freeze > requirements-lock.txt` and install from the
lock file in the Dockerfile.

### 3. Use .dockerignore aggressively

Every file not in `.dockerignore` is sent to the Docker daemon as build context. Our
project excludes `data/raw/` (potentially 150 MB), `mlruns/` (potentially gigabytes),
and `.git/` (potentially hundreds of MB). Without `.dockerignore`, a `docker build` on
our project would send 500+ MB of unnecessary files to the daemon, slowing down every build.

### 4. Combine RUN commands to reduce layers

```dockerfile
# GOOD: one layer for all pip operations
RUN pip install --no-cache-dir -r requirements.txt

# BAD: multiple layers for sequential operations
RUN pip install numpy
RUN pip install pandas
RUN pip install xgboost
```

Each `RUN` creates a new layer. Fewer layers = smaller image (in some cases) and faster
pulls. However, do not combine unrelated operations (like pip install and file copies)
because that defeats layer caching.

### 5. Clean up in the same layer

```dockerfile
# GOOD: cleanup in the same RUN command
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# BAD: cleanup in a separate layer (original files still in previous layer)
RUN pip install -r requirements.txt
RUN rm -rf /tmp/*
```

Docker layers are additive. Deleting files in a new layer does not reduce the image size
because the files still exist in the previous layer. You must clean up in the same `RUN`
command.

### 6. Use slim or Alpine base images (when not using Lambda)

```
Image size comparison (approximate):
  python:3.12          ~1.0 GB
  python:3.12-slim     ~150 MB
  python:3.12-alpine   ~50 MB
  lambda/python:3.12   ~300 MB (what we use)
```

Alpine is the smallest but can cause problems with Python packages that have C extensions
(NumPy, XGBoost) because Alpine uses musl libc instead of glibc. For ML workloads,
`python:3.12-slim` is the best non-Lambda option.

---

## Local Development with Docker

Our `Makefile` provides convenient Docker commands:

```bash
# Build the Docker image
make docker-build
# Equivalent to: docker build -t mlops-fraud-detection:latest .

# Run the container locally (for testing the FastAPI app)
make docker-run
# Equivalent to: docker run -p 8000:8000 mlops-fraud-detection:latest
```

For local development without Lambda, you can also use Docker Compose (not currently in
our project but a natural extension):

```yaml
# docker-compose.yml (example for local development)
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
    command: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

  mlflow:
    image: python:3.12-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: pip install mlflow && mlflow ui --host 0.0.0.0 --port 5000
```

---

## Container Security for ML

### 1. Never include credentials in the image

```dockerfile
# WRONG: copying AWS credentials into the image
COPY AWS_Credential /root/.aws/credentials

# RIGHT: credentials come from IAM roles (Lambda) or environment variables
# Our Lambda function gets permissions from its IAM role (lambda.tf)
```

Our `.dockerignore` explicitly excludes `AWS_Credential` and `.env` to prevent accidental
inclusion. The pre-commit hook `detect-private-key` provides an additional safety net.

### 2. Scan images for vulnerabilities

ECR's `scan_on_push = true` automatically scans every pushed image. You can also scan
locally before pushing:

```bash
# Using Docker Scout (built into Docker Desktop)
docker scout cves mlops-fraud-detection:latest

# Using Trivy (open source)
trivy image mlops-fraud-detection:latest
```

### 3. Use non-root users (for non-Lambda images)

```dockerfile
# For non-Lambda deployments, add a non-root user
RUN useradd -m -s /bin/bash appuser
USER appuser
```

Lambda containers run as a non-root user by default, so this is not needed for our
current setup.

### 4. Minimal attack surface

Our `.dockerignore` keeps tests, documentation, infrastructure code, and development
tools out of the production image. The less code in the container, the smaller the
attack surface.

---

## Interview Questions and Answers

### Q1: Why do you use Docker for ML model deployment?

**A:** Docker ensures that the exact same environment used for training is used for
inference. Without Docker, a model trained on one machine can produce different results on
another due to library version differences, system library differences, or Python version
differences. Docker packages the Python runtime, all dependencies (scikit-learn, XGBoost,
NumPy), system libraries, the model artifact, and the serving code into a single immutable
unit. This guarantees reproducibility and eliminates "it works on my machine" problems.

### Q2: Walk me through your Dockerfile and explain each decision.

**A:** We use the AWS Lambda Python 3.12 base image because our model is deployed as a
Lambda function. We copy `requirements.txt` first and install dependencies before copying
code --- this leverages Docker layer caching so dependency installation (the slowest step)
is skipped when only code changes. We copy `src/`, `configs/`, and `params.yaml` for the
application code. We copy `models/` to bake the trained model into the image. The CMD
points to the Mangum handler that wraps our FastAPI app for Lambda invocation. We use
`--no-cache-dir` with pip to reduce image size. We exclude unnecessary files via
`.dockerignore` (tests, docs, raw data, credentials).

### Q3: What is the difference between baking the model into the image vs. loading it at runtime from S3?

**A:** Baking the model in (our approach) means the model is part of the immutable image.
Pros: faster cold starts (no S3 download), no runtime dependency on S3, guaranteed
consistency between code and model. Cons: changing the model requires a new image and
redeployment. Loading from S3 at runtime means the image is model-agnostic. Pros: can
swap models without redeploying, one image serves multiple model versions. Cons: slower
cold starts (Lambda must download from S3 on first invocation), requires S3 permissions,
risk of model/code version mismatch. For our use case (fraud detection, sub-second latency
required, infrequent retraining), baking in is the right choice.

### Q4: How does Docker layer caching work and why does it matter for ML?

**A:** Docker builds images in layers, one per instruction. If a layer's inputs have not
changed, Docker reuses the cached layer. Layers are checked top-to-bottom; once a layer
cache is invalidated, all subsequent layers are rebuilt. For ML, this matters because
`pip install` is slow (2-5 minutes for our dependencies). By copying `requirements.txt`
and running pip install BEFORE copying code, we ensure dependency installation is cached
whenever only code changes. If we copied everything at once, any code change would trigger
a full dependency reinstall.

### Q5: What is the purpose of .dockerignore and what should you exclude for ML projects?

**A:** `.dockerignore` specifies files that should not be sent to the Docker daemon during
builds. For ML projects, you should exclude: (1) Training data (`data/raw/`) --- not needed
for inference and can be huge. (2) MLflow data (`mlruns/`) --- experiment tracking data
can be gigabytes. (3) Notebooks (`notebooks/`) --- development artifacts. (4) Tests
(`tests/`) --- not needed at runtime. (5) Credentials (`.env`, `AWS_Credential`) --- never
include secrets in images. (6) Git history (`.git/`) --- can be hundreds of MB and is
irrelevant at runtime. Without `.dockerignore`, our build context would be 500+ MB instead
of ~60 MB.

### Q6: How do you handle model versioning with Docker?

**A:** Each Docker image is tagged with the Git commit SHA that produced it (e.g.,
`$ECR_REGISTRY/mlops-fraud-detection:abc123`). Since the model is baked into the image,
the image tag effectively versions the model. To find which model is in production, check
the Lambda function's image URI. To rollback, update Lambda to use a previous image tag.
ECR retains the last 10 images per our lifecycle policy, providing a window for rollbacks.
For more sophisticated model versioning, you could add labels to the Docker image with
the MLflow run ID and model metrics.

### Q7: What are Lambda container cold starts and how do you minimize them?

**A:** A cold start occurs when Lambda creates a new instance of your container to handle
a request. During a cold start, Lambda must: (1) Pull the container image from ECR (first
time only, then cached). (2) Start the container. (3) Initialize the runtime. (4) Run the
handler's initialization code (for us: load the model from disk). This can take 5-15 seconds
for ML containers. To minimize cold starts: (1) Use provisioned concurrency (keeps instances
warm). (2) Minimize image size (faster pulls). (3) Bake the model into the image (no S3
download). (4) Use lazy loading for rarely-used dependencies. (5) Keep the container memory
at 512 MB or more (Lambda allocates CPU proportional to memory). Our FastAPI app uses the
`lifespan` context manager to load the model at container startup, so the model is already
in memory when the first request arrives.

### Q8: How would you deploy this model to Kubernetes instead of Lambda?

**A:** The main changes would be: (1) Replace the Lambda base image with `python:3.12-slim`.
(2) Replace the CMD with `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`. (3)
Remove the Mangum handler (FastAPI serves directly). (4) Add a Kubernetes Deployment and
Service manifest. (5) Use a multi-stage Docker build to reduce image size. (6) Set up a
Horizontal Pod Autoscaler based on CPU/request count. (7) Use Kubernetes readiness/liveness
probes pointing to our `/health` endpoint. (8) Consider mounting the model from a PersistentVolume
or init container instead of baking it in, for faster model updates without image rebuilds.
The FastAPI app code (`src/serving/app.py`) would not change at all --- it is framework-agnostic.

### Q9: Explain the security considerations for your Docker deployment.

**A:** (1) **No credentials in images.** `.dockerignore` excludes `AWS_Credential` and `.env`.
Lambda gets permissions from its IAM role. (2) **Vulnerability scanning.** ECR scans every
pushed image for known CVEs. (3) **Minimal image contents.** Tests, docs, raw data, and
development tools are excluded, reducing attack surface. (4) **Encrypted storage.** ECR
images are encrypted at rest with AES-256. Our S3 bucket also uses server-side encryption.
(5) **Private registry.** ECR is private by default; only our AWS account can pull images.
(6) **Image lifecycle.** Old images are automatically expired, reducing the window for
exploiting outdated dependencies. (7) **Pre-commit hooks.** `detect-private-key` and
`gitleaks` prevent credentials from entering the codebase and potentially the Docker image.

---

## Practical Tips

1. **Always use `.dockerignore`.** For ML projects, the build context without `.dockerignore`
   can easily be 1+ GB (training data, MLflow runs, Git history). With `.dockerignore`, our
   build context is ~60 MB.

2. **Copy requirements.txt before code.** This is the single most impactful layer caching
   optimization. For our project, pip install takes ~2 minutes. With this optimization, code
   changes rebuild in ~10 seconds.

3. **Use `--no-cache-dir` with pip.** This saves 50-100 MB of image space by not caching
   downloaded packages. The cache is useless in a container because you never run pip again
   inside the production container.

4. **Tag images with Git SHA, not just `latest`.** The SHA tag creates an immutable,
   traceable link between a deployed container and the exact code that built it. Our deploy
   workflow pushes both `$GITHUB_SHA` and `latest` tags.

5. **Test your container locally before pushing.** Use `make docker-build && make docker-run`
   and hit `http://localhost:8000/health` before triggering the deployment pipeline.

6. **Monitor image size over time.** Our image is ~810 MB. If it starts growing
   significantly, investigate which dependencies are adding weight. Use `docker history
   <image>` to see layer sizes. Consider switching to slim base images or removing unused
   dependencies.

7. **Set appropriate Lambda resource limits.** Our Lambda function uses 512 MB memory and
   60-second timeout. XGBoost inference is fast (< 100ms) and memory-efficient, so these
   limits work well. If you switch to a larger model (e.g., deep learning), you may need
   to increase memory to 1-3 GB.
