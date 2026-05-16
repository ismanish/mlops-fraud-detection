# 06 — Model Serving on AWS

## Table of Contents
1. [Serving Patterns Overview](#serving-patterns-overview)
2. [Our Architecture: FastAPI + Lambda + API Gateway](#our-architecture)
3. [Lambda Container Images Explained](#lambda-container-images)
4. [Mangum: Bridging FastAPI to Lambda](#mangum-bridging-fastapi-to-lambda)
5. [API Gateway Deep Dive](#api-gateway-deep-dive)
6. [Cold Starts and Mitigation](#cold-starts-and-mitigation)
7. [Cost Comparison: Lambda vs ECS vs SageMaker vs EC2](#cost-comparison)
8. [Our FastAPI App Walkthrough](#fastapi-app-walkthrough)
9. [Interview Questions](#interview-questions)

---

## Serving Patterns Overview

There are three fundamental patterns for serving ML models in production. Each suits
different latency, throughput, and cost profiles.

### Batch Inference

Process large volumes of data on a schedule (hourly, daily). Results are written to a
database or data lake, and downstream systems read pre-computed predictions.

```
+------------+       +-----------+       +-----------+       +----------+
| Data Lake  | ----> | Batch Job | ----> |  Model    | ----> | Results  |
| (S3/BQ)    |       | (Spark /  |       | .predict  |       | Table /  |
|            |       |  Airflow) |       |  _batch() |       | S3       |
+------------+       +-----------+       +-----------+       +----------+
```

**When to use:** Fraud reports generated overnight, recommendation pre-computation,
churn scoring for marketing campaigns. Latency tolerance is minutes to hours.

**Our project supports this** via `predict_batch()` in `src/models/predict.py`:

```python
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if _model is None:
        load_artifacts()
    features = df.copy()
    if "Amount" in features.columns and _scaler is not None:
        features["Amount"] = _scaler.transform(features[["Amount"]])
    if "Time" in features.columns:
        features = features.drop(columns=["Time"])
    probabilities = _model.predict_proba(features)[:, 1]
    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = (probabilities >= 0.5).astype(int)
    return result
```

### Real-Time (Online) Inference

The client sends a single request, the model runs synchronously, and a prediction is
returned within milliseconds. This is what our project implements.

```
+---------+       +---------------+       +--------+       +---------+
| Client  | ----> | API Gateway   | ----> | Lambda | ----> | XGBoost |
| (POST   |       | (HTTP API)    |       | (Fast  |       | .predict|
|  /predict)      |               |       |  API)  |       |  _proba |
+---------+       +---------------+       +--------+       +---------+
     ^                                         |
     |                                         |
     +---- JSON { prediction, probability } ---+
```

**When to use:** Transaction-time fraud detection (our case), content moderation at
upload time, real-time pricing. Latency requirement is under 100-500ms.

### Streaming Inference

Model inference sits inside a streaming pipeline (Kafka, Kinesis, Flink). Events
flow continuously and predictions are emitted to a downstream topic.

```
+---------+       +----------+       +----------+       +-----------+
| Kafka   | ----> | Flink /  | ----> | Model    | ----> | Kafka     |
| Topic:  |       | Kinesis  |       | predict  |       | Topic:    |
| txn_raw |       | Consumer |       |          |       | txn_scored|
+---------+       +----------+       +----------+       +-----------+
```

**When to use:** High-throughput event streams (millions of events/second), IoT
sensor data, click-stream analysis. Combines low latency with high throughput.

### Pattern Comparison Table

```
+-------------+-----------+------------+------------+-----------+
| Pattern     | Latency   | Throughput  | Complexity | Cost      |
+-------------+-----------+------------+------------+-----------+
| Batch       | Minutes+  | Very High   | Low        | Low       |
| Real-Time   | <500ms    | Medium      | Medium     | Medium    |
| Streaming   | <100ms    | Very High   | High       | High      |
+-------------+-----------+------------+------------+-----------+
```

---

## Our Architecture

Our fraud detection system uses a serverless real-time serving stack:

```
                         Internet
                            |
                            v
                  +-------------------+
                  |   API Gateway     |
                  |   (HTTP API)      |
                  |  - CORS config    |
                  |  - Access logs    |
                  |  - Throttling     |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |   AWS Lambda      |
                  |   (Container)     |
                  |                   |
                  |  +--------------+ |
                  |  |   Mangum     | |
                  |  |  (ASGI      | |
                  |  |   adapter)  | |
                  |  +------+------+ |
                  |         |        |
                  |  +------v------+ |
                  |  |  FastAPI    | |
                  |  |  - /health  | |
                  |  |  - /predict | |
                  |  +------+------+ |
                  |         |        |
                  |  +------v------+ |
                  |  |  XGBoost   | |
                  |  |  model.pkl | |
                  |  +-------------+ |
                  +--------+---------+
                           |
                     +-----+-----+
                     |           |
                     v           v
              +-----------+ +------------+
              | CloudWatch| | S3 Bucket  |
              | Metrics   | | (artifacts)|
              +-----------+ +------------+
```

### Why This Stack?

1. **Serverless** -- No servers to manage, automatic scaling from 0 to thousands of
   concurrent requests.
2. **Pay-per-use** -- Zero cost when idle. For a learning/portfolio project, this
   means near-zero AWS bills.
3. **FastAPI** -- Modern Python framework with automatic OpenAPI docs, Pydantic
   validation, and async support. Reusable locally during development.
4. **Container Lambda** -- Our Docker image bundles the model, scaler, code, and all
   dependencies. No 50MB zip file limit; container images support up to 10GB.

---

## Lambda Container Images

Traditional Lambda deployment packages are zip files with a 50MB limit (250MB
uncompressed). ML models with scikit-learn, XGBoost, and NumPy easily exceed this.
Container image support solves this problem.

### How It Works

Lambda container images must implement the Lambda Runtime Interface. AWS provides
base images that include the Runtime Interface Client (RIC) and Runtime Interface
Emulator (RIE).

Our `Dockerfile`:

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

**Key details:**

| Line | What It Does |
|------|-------------|
| `FROM public.ecr.aws/lambda/python:3.12` | AWS-maintained base image with the Lambda runtime pre-installed |
| `${LAMBDA_TASK_ROOT}` | Environment variable set to `/var/task` -- the working directory Lambda expects |
| `COPY models/` | The trained `model.pkl` is baked into the image at build time |
| `CMD [...]` | Points to the Mangum handler entry point |

### Image Size Considerations

```
Layer Breakdown (approximate):
  Base image (python:3.12)        ~280 MB
  pip install requirements.txt    ~350 MB   (XGBoost, scikit-learn, pandas, etc.)
  Source code                     ~  5 MB
  Model artifacts (model.pkl)     ~  3 MB   (XGBoost models are compact)
  ----------------------------------------
  Total compressed                ~350 MB   (layers are compressed in ECR)
```

### Build and Deploy Flow

```
Developer Machine               ECR                     Lambda
+----------------+      +----------------+      +------------------+
| docker build   | ---> | docker push    | ---> | update-function  |
| -t repo:tag .  |      | repo:tag       |      | -code --image-uri|
+----------------+      +----------------+      +------------------+
```

This is automated in `scripts/deploy.sh` and `.github/workflows/deploy.yml`.

---

## Mangum: Bridging FastAPI to Lambda

Lambda expects a handler function with the signature `handler(event, context)`.
FastAPI is an ASGI application. Mangum translates between these two worlds.

### Our Lambda Handler

```python
# src/serving/lambda_handler.py
from mangum import Mangum
from src.serving.app import app

handler = Mangum(app, lifespan="on")
```

That is the entire file. Three lines.

### What Mangum Does Internally

```
API Gateway Event (JSON)          Lambda Response (JSON)
+------------------------+       +------------------------+
| {                      |       | {                      |
|   "httpMethod": "POST",| ----->|   "statusCode": 200,   |
|   "path": "/predict",  |       |   "body": "{...}",     |
|   "body": "{...}",     |       |   "headers": {...}     |
|   "headers": {...}     |       | }                      |
| }                      |       +------------------------+
+----------+-------------+                ^
           |                              |
           v                              |
+----------+-------------+               |
|       Mangum           |               |
| 1. Parse event         |               |
| 2. Build ASGI scope    |               |
| 3. Call FastAPI app    +---------------+
| 4. Capture response    |
| 5. Format Lambda reply |
+------------------------+
```

### The `lifespan="on"` Parameter

This is critical. It tells Mangum to execute FastAPI's lifespan events. Our app
uses the lifespan context manager to load the model at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()        # Load model.pkl and scaler.pkl into memory
    logger.info("Model loaded -- ready to serve")
    yield                   # App runs; model is warm in memory
```

Without `lifespan="on"`, the model would never be loaded, and every prediction
request would fail.

### Why Not Just Use Lambda Directly?

You could write a raw Lambda handler. But then you lose:

- **Pydantic validation** -- FastAPI validates every field of the request body
  automatically and returns clear 422 errors for bad input.
- **Auto-generated OpenAPI docs** -- Visit `/docs` in local development for
  interactive API testing.
- **Local development** -- Run `uvicorn src.serving.app:app --reload` and test
  without deploying to AWS.
- **Middleware ecosystem** -- CORS, rate limiting, authentication middleware.
- **Portability** -- The same FastAPI app deploys to Lambda, ECS, Kubernetes, or
  a bare EC2 instance with zero code changes.

---

## API Gateway Deep Dive

API Gateway is the front door to our Lambda function. It handles routing, throttling,
CORS, and access logging.

### HTTP API vs REST API

AWS offers two API Gateway products. We use HTTP API.

```
+-------------------+-------------------+---------------------+
| Feature           | HTTP API          | REST API            |
+-------------------+-------------------+---------------------+
| Latency           | ~10ms overhead    | ~30ms overhead      |
| Cost (per million)| $1.00             | $3.50               |
| WebSocket         | No                | Yes                 |
| API Keys          | No (use Lambda    | Built-in            |
|                   |  authorizer)      |                     |
| Usage Plans       | No                | Yes                 |
| Caching           | No                | Built-in            |
| Request transform | No                | Velocity templates  |
| WAF integration   | No (use CloudFront| Direct              |
|                   |  in front)        |                     |
| CORS              | Built-in toggle   | Manual headers      |
+-------------------+-------------------+---------------------+
```

**Our choice: HTTP API** because it is cheaper, lower latency, and sufficient for
our REST-only use case.

### Our Terraform Configuration

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
}
```

**CORS configuration** allows any origin to call our API. In production, you would
restrict `allow_origins` to your frontend domain.

### Route Configuration

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

Two routes: `POST /predict` for inference and `GET /health` for health checks.
Both route to the same Lambda function; Mangum/FastAPI handles internal routing.

### Throttling

HTTP API has default throttling of 10,000 requests/second per account per region.
You can configure route-level throttling:

```
Default throttle:  10,000 req/s (burst), 5,000 req/s (rate)
Route throttle:    Configurable per route
Account limit:     Soft limit, can request increase
```

For fraud detection, this is typically more than sufficient. A large bank might
process 1,000-5,000 transactions per second.

---

## Cold Starts and Mitigation

A cold start occurs when Lambda must initialize a new execution environment. This
includes downloading the container image, starting the runtime, and running
initialization code (our model loading).

### Cold Start Timeline

```
                         Cold Start (~3-8 seconds)
     |<------------------------------------------------------>|
     |                                                         |
     | Download   | Start     | Init     | Handler  |          |
     | Container  | Runtime   | Code     | Execution| Response |
     | Image      |           | (load    |          |          |
     | (~1-3s)    | (~0.5s)   | model)   | (~50ms)  |          |
     |            |           | (~1-2s)  |          |          |
     |<---------->|<--------->|<-------->|<-------->|          |

                         Warm Invocation (~50-100ms)
     |<----------->|
     |             |
     | Handler     | Response
     | Execution   |
     | (~50ms)     |
```

### Why Cold Starts Hurt for ML

Our model loading step (`load_artifacts()`) deserializes `model.pkl` and
`scaler.pkl` from disk. For XGBoost, this is fast (~200ms), but for larger models
(deep learning, large ensembles), it can take seconds.

### Mitigation Strategies

**1. Provisioned Concurrency** (our recommended approach for production)
```
aws lambda put-provisioned-concurrency-config \
    --function-name mlops-fraud-detection-predict \
    --qualifier production \
    --provisioned-concurrent-executions 5
```
Keeps 5 warm instances ready at all times. Cost: ~$15/month per instance.

**2. Smaller Container Images**
- Use slim base images
- Remove unnecessary dependencies
- Multi-stage Docker builds

**3. Model Optimization**
- Use ONNX Runtime instead of full XGBoost (faster deserialization)
- Quantize the model to reduce file size
- Load model from `/tmp` (Lambda's ephemeral storage) after first download

**4. Lambda SnapStart** (Java only as of 2025, not applicable to Python)

**5. Keep-Warm Pings**
Schedule a CloudWatch Events rule to invoke the Lambda every 5 minutes:
```
+------------------+       +---------+       +--------+
| CloudWatch Rule  | ----> | Lambda  | ----> | /health|
| (every 5 min)    |       |         |       | (warm) |
+------------------+       +---------+       +--------+
```
Cheap but unreliable -- Lambda can still reclaim the instance.

**6. Move to ECS/Fargate**
If cold starts are unacceptable, run the FastAPI app as an always-on container.
See cost comparison below.

---

## Cost Comparison

Estimated costs for serving 100,000 predictions/day (~1.2 req/sec average).

```
+-------------------+------------+---------------+-------------------+
| Service           | Monthly $  | Cold Start    | Scaling           |
+-------------------+------------+---------------+-------------------+
| Lambda            |   $3-5     | 3-8s (first)  | Auto (0 to 1000s) |
| Lambda + Prov.    |   $20-80   | None          | Auto + reserved   |
| ECS Fargate       |   $30-50   | None          | Task-based (min 1)|
| SageMaker Endpt   |   $50-150  | None          | Auto (min 1 inst) |
| EC2 (t3.medium)   |   $30      | None          | Manual / ASG      |
+-------------------+------------+---------------+-------------------+
```

### Lambda (our choice)

```
Compute:  100K requests x 512MB x 200ms avg = 10,000 GB-seconds
          Free tier: 400,000 GB-seconds/month
          Cost: $0 (within free tier for modest traffic)

Requests: 100K x 30 days = 3M requests
          Free tier: 1M requests/month
          Cost: 2M x $0.20/M = $0.40

API GW:   3M requests x $1.00/M = $3.00
          Total: ~$3.40/month
```

### ECS Fargate

```
1 task x 0.5 vCPU x 1 GB memory x 730 hours
vCPU:   0.5 x $0.04048/hr x 730 = $14.78
Memory: 1 x $0.004445/hr x 730 = $3.24
ALB:    $16.20 (fixed) + LCU costs
Total: ~$35-50/month
```

### SageMaker Real-Time Endpoint

```
ml.m5.large: $0.115/hr x 730 = $83.95/month
Inference:   $0.0002/request overhead
Total: ~$85-100/month (minimum, single instance)
```

### When to Choose What

| Scenario | Best Choice |
|----------|------------|
| Portfolio project, low traffic | Lambda (our choice) |
| Production, <1000 req/s, latency-sensitive | Lambda + Provisioned Concurrency |
| Production, >1000 req/s, sustained load | ECS Fargate or EKS |
| Need A/B testing, shadow deployment | SageMaker Endpoints |
| Need GPU inference (deep learning) | SageMaker or EC2 with GPU |
| Maximum cost control, predictable load | EC2 with ASG |

---

## FastAPI App Walkthrough

### File: `src/serving/app.py`

#### Application Initialization

```python
app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Real-time credit card fraud detection",
    lifespan=lifespan,
)
```

The `lifespan` parameter registers startup/shutdown logic. On startup, the model
and scaler are loaded into module-level globals via `load_artifacts()`.

#### Request Model (Pydantic)

```python
class TransactionRequest(BaseModel):
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    # ... V3 through V28 ...
    Amount: float = Field(..., ge=0, description="Transaction amount")
```

Key design decisions:
- **28 PCA features (V1-V28)** -- These are the anonymized features from the original
  Kaggle credit card dataset. PCA was applied by the dataset authors for privacy.
- **Amount field with `ge=0`** -- Pydantic enforces that Amount must be non-negative.
  A request with `Amount: -5.0` returns a 422 Validation Error automatically.
- **No Time field** -- Time is dropped during preprocessing (`features.drop_columns`
  in params.yaml), so the API does not accept it.

#### Response Model

```python
class PredictionResponse(BaseModel):
    prediction: int          # 0 or 1
    fraud_probability: float # 0.0 to 1.0
    label: str               # "fraud" or "legitimate"
    latency_ms: float        # End-to-end prediction time
```

Including `latency_ms` in the response is a production best practice. It lets clients
log and monitor prediction latency from the model's perspective (excluding network).

#### Health Check Endpoint

```python
@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}
```

Used by:
- API Gateway health checks
- Load balancers (if migrating to ECS)
- The deploy workflow's smoke test
- Monitoring dashboards

A more robust health check would verify the model is actually loaded:

```python
@app.get("/health")
def health():
    from src.models.predict import _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}
```

#### Prediction Endpoint

```python
@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    start = time.time()
    try:
        result = predict(transaction.model_dump())
        latency = (time.time() - start) * 1000
        result["latency_ms"] = round(latency, 2)
        _publish_metrics(result)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

The flow:
1. `transaction.model_dump()` converts Pydantic model to a plain dict
2. `predict()` scales the Amount field, runs XGBoost inference, returns prediction
3. Latency is measured and appended to the response
4. CloudWatch metrics are published asynchronously (best-effort, failures logged)
5. The result is validated against `PredictionResponse` and returned as JSON

#### CloudWatch Metrics Publishing

```python
def _publish_metrics(result: dict):
    cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
    cloudwatch.put_metric_data(
        Namespace="MLOps/FraudDetection",
        MetricData=[
            {"MetricName": "PredictionLatency", "Value": result["latency_ms"],
             "Unit": "Milliseconds"},
            {"MetricName": "FraudPredicted", "Value": result["prediction"],
             "Unit": "Count"},
            {"MetricName": "FraudProbability", "Value": result["fraud_probability"],
             "Unit": "None"},
        ],
    )
```

Three metrics published per prediction:
- **PredictionLatency** -- Tracks P50/P95/P99 latency
- **FraudPredicted** -- Sum over time = fraud count; useful for anomaly detection
- **FraudProbability** -- Average over time tracks prediction distribution shift

### Example Request/Response

```bash
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
    "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
    "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
    "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
    "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
    "V28": -0.021053, "Amount": 149.62
  }'

# Response:
{
  "prediction": 0,
  "fraud_probability": 0.023456,
  "label": "legitimate",
  "latency_ms": 12.34
}
```

---

## Interview Questions

### Q1: Why did you choose Lambda over SageMaker for model serving?

**A:** Our XGBoost model is lightweight (~3MB) with sub-100ms inference time, so we
do not need SageMaker's built-in model hosting features like automatic A/B testing or
multi-model endpoints. Lambda with container images gives us serverless scaling,
pay-per-invocation pricing (near-zero cost at low traffic), and the flexibility to
run the same FastAPI code locally during development. SageMaker would cost $84+/month
minimum for a single always-on endpoint, whereas Lambda stays within free tier for
our traffic levels. If we needed GPU inference or built-in model monitoring,
SageMaker would be the better choice.

### Q2: How does Mangum work, and why is it necessary?

**A:** Mangum is an ASGI adapter that translates AWS Lambda's `event`/`context`
invocation format into the ASGI protocol that FastAPI expects. When API Gateway
sends a request to Lambda, the event is a JSON object with `httpMethod`, `path`,
`body`, and `headers`. Mangum converts this into an ASGI scope, calls the FastAPI
app, captures the response, and reformats it into Lambda's expected return format
(`statusCode`, `body`, `headers`). Without Mangum, we would need to manually parse
API Gateway events and lose all of FastAPI's features like Pydantic validation,
middleware, and auto-generated documentation.

### Q3: What happens during a Lambda cold start, and how would you mitigate it?

**A:** A cold start occurs when Lambda must create a new execution environment: it
downloads the container image from ECR, starts the Python runtime, imports
dependencies, and runs our `lifespan` handler which loads the model from disk. For
our setup, this takes 3-8 seconds. Mitigation strategies include Provisioned
Concurrency (keeps N warm instances), keep-warm pings via CloudWatch scheduled
events, reducing container image size with multi-stage builds, and using lighter
serialization formats like ONNX. For production fraud detection where latency is
critical, I would use Provisioned Concurrency with 5-10 instances and consider
migrating to ECS Fargate if sustained throughput exceeds Lambda's cost-effectiveness.

### Q4: Explain the difference between HTTP API and REST API in API Gateway.

**A:** HTTP API (what we use) is newer, cheaper ($1/M vs $3.50/M requests), lower
latency (~10ms vs ~30ms overhead), and simpler. It supports JWT authorizers, Lambda
integration, and automatic CORS configuration. REST API offers additional features
like API key management, usage plans with throttling quotas, request/response
transformation via Velocity templates, built-in caching, and direct WAF integration.
For a pure Lambda-backed REST API without caching needs, HTTP API is the clear
choice. If we needed WebSocket support, API key-based access control, or response
caching at the gateway level, we would use REST API.

### Q5: How would you handle model versioning in this serving setup?

**A:** Each Docker image in ECR is tagged with the git commit SHA
(`$ECR_REGISTRY/$ECR_REPOSITORY:$GITHUB_SHA`). This means every deployed model
version is traceable to a specific code commit. For canary deployments, I would use
Lambda aliases with weighted routing -- for example, 90% traffic to the `production`
alias and 10% to `canary`. Lambda aliases can point to different image versions. For
A/B testing, I would use API Gateway stage variables to route different paths to
different Lambda versions. The model itself is tracked in MLflow with a run_id, so we
have full lineage from prediction back to training data and hyperparameters.

### Q6: Why do you publish CloudWatch metrics from the application code?

**A:** Lambda automatically publishes invocation count, duration, and error count.
But these are infrastructure metrics -- they do not capture ML-specific signals. We
publish custom metrics (PredictionLatency, FraudPredicted, FraudProbability) because
they enable ML-aware monitoring: tracking prediction distribution shift over time,
detecting if fraud rate suddenly spikes (potential attack or model degradation), and
measuring model-level latency separately from Lambda overhead. These custom metrics
feed into our CloudWatch dashboard and can trigger alarms -- for example, if the
average FraudProbability rises above 0.3, it might indicate the model is
miscalibrated or the input distribution has shifted.

### Q7: Your _publish_metrics function is synchronous. Is that a problem?

**A:** Yes, it adds latency to every prediction request because the CloudWatch
`put_metric_data` API call takes 5-20ms. In production, I would address this by
using a fire-and-forget pattern: either publish metrics asynchronously using
`asyncio.create_task()` (since FastAPI supports async), batch metrics using the
CloudWatch Embedded Metrics Format (EMF) which writes to stdout and is automatically
parsed by CloudWatch, or use a side-channel like writing to a Kinesis Data Firehose
stream. EMF is the most elegant solution for Lambda because it has zero API call
overhead -- you just print structured JSON to stdout.

### Q8: How would you scale this architecture to 10,000 requests per second?

**A:** Lambda scales automatically to the account's concurrency limit (default 1,000
concurrent executions, can request increase to 10,000+). At 10K RPS with 100ms
average duration, we need ~1,000 concurrent executions -- right at the default limit.
I would request a concurrency increase, enable Provisioned Concurrency for 200-500
instances to eliminate cold starts, and monitor the `Throttles` CloudWatch metric.
If sustained at 10K RPS, I would evaluate migrating to ECS Fargate with horizontal
auto-scaling, since Lambda becomes more expensive than containers at high sustained
throughput. The breakeven point is typically around 1-2 million requests/day.
