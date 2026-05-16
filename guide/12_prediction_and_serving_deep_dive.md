# 12 --- Prediction and Serving Deep Dive

## Table of Contents
1. [Prediction Module (predict.py) --- Line by Line](#prediction-module-predictpy--line-by-line)
2. [FastAPI Application (app.py) --- Line by Line](#fastapi-application-apppy--line-by-line)
3. [Lambda Handler (lambda_handler.py) --- Line by Line](#lambda-handler-lambda_handlerpy--line-by-line)
4. [API Design Patterns for ML](#api-design-patterns-for-ml)
5. [Interview Questions](#interview-questions)

---

## Prediction Module (predict.py) --- Line by Line

This is the core inference engine. Everything else --- the FastAPI server, the Lambda handler,
the batch job --- is just a wrapper around this module. Understanding it deeply means
understanding how ML models are served in production everywhere from Stripe to Netflix.

### The Imports

```python
"""Prediction module: loads model and returns fraud probability."""

import joblib
import pandas as pd

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

**`joblib`** is the standard serialization library for scikit-learn models. It is built on
top of Python's `pickle` but is optimized for objects that carry large numpy arrays internally.
When you call `joblib.dump(model, "model.pkl")` during training, it serializes the entire
fitted model --- the learned weights, the tree structure, the hyperparameters, everything ---
into a binary file. `joblib.load` reverses that process. The `.pkl` extension stands for
"pickle," the Python serialization format.

Why joblib over plain pickle? A trained Random Forest with 200 trees contains thousands of
numpy arrays. Plain pickle serializes each one byte-by-byte. Joblib uses numpy's internal
save mechanism, which is dramatically faster for large arrays. In benchmarks, joblib loads a
200-tree Random Forest in roughly 200ms versus 1.5 seconds for raw pickle.

**`pandas`** is imported because the model was trained on a pandas DataFrame, so prediction
inputs must match that format. Scikit-learn models remember the feature order they were
trained with. If you pass a numpy array with columns in a different order, predictions will
be silently wrong --- one of the most insidious bugs in production ML.

**`get_project_root()`** returns the absolute path to the project root directory. This matters
because the code might be called from different working directories (a test runner, a Docker
container, a Lambda function). Using an absolute project root makes file paths reliable
regardless of where the code executes.

**`get_logger(__name__)`** creates a logger whose name is the fully-qualified module name,
`src.models.predict`. This means log lines will identify exactly which module emitted them,
critical when debugging a system with dozens of modules.

---

### The Global State: Singleton Pattern for ML Models

```python
_model = None
_scaler = None
```

These two lines implement the **Singleton pattern** for ML artifacts. The underscore prefix
(`_model`, not `model`) is a Python convention signaling "this is private to this module ---
do not import or touch it from outside."

**Why global state?** Loading an ML model from disk is expensive. Our Random Forest model is
roughly 50-100MB on disk. Deserializing it with joblib takes 200-500ms. If we loaded the
model fresh on every prediction request, a server handling 100 requests/second would spend
all its time loading models, not making predictions.

By storing the model in a module-level global, we load it once and reuse it for every
subsequent call. This is the same pattern used at:

- **Stripe** --- their fraud models are loaded once into each server process and shared across
  all request-handling threads
- **Netflix** --- recommendation models are loaded into memory at container startup and
  persisted for the container's lifetime
- **Uber** --- Michelangelo (their ML platform) preloads models into serving containers

The pattern is sometimes called "module-level caching" or "process-level caching." It works
because Python module globals persist for the lifetime of the process.

**Why `None` and not loaded immediately?** This is the **lazy loading** pattern. The model is
not loaded when the module is imported --- it is loaded only when the first prediction is
requested. This has several benefits:

1. **Faster imports.** If you run `import src.models.predict` in a test that never calls
   `predict()`, you do not pay the 500ms model-loading cost.
2. **Testability.** Tests can mock `_model` and `_scaler` without needing real model files.
3. **Flexibility.** The caller can choose when to load by calling `load_artifacts()` explicitly
   (as the FastAPI lifespan does) or let it happen automatically on first use.

---

### `load_artifacts()` --- Lazy Loading Pattern

```python
def load_artifacts(model_path: str | None = None, scaler_path: str | None = None):
    global _model, _scaler
    root = get_project_root()

    if model_path is None:
        model_path = str(root / "models" / "model.pkl")
    if scaler_path is None:
        scaler_path = str(root / "data" / "processed" / "scaler.pkl")

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded")
```

**`global _model, _scaler`** --- This line is required because the function assigns to
`_model` and `_scaler`. Without the `global` declaration, Python would treat them as local
variables, and the module-level `_model` would remain `None` after the function returns. This
is a common Python gotcha: reading a global variable works without `global`, but writing to
it requires the declaration.

**`str | None = None`** --- The type hint uses Python 3.10+ union syntax. `str | None` means
the parameter accepts either a string or None. The `= None` default means callers can omit
the argument entirely. This gives us two usage patterns:

```python
# Pattern 1: Use default paths (production)
load_artifacts()

# Pattern 2: Specify custom paths (testing, different model versions)
load_artifacts(model_path="/opt/models/v2/model.pkl")
```

**`root / "models" / "model.pkl"`** --- The `/` operator on `Path` objects concatenates path
segments. This is cleaner than `os.path.join(root, "models", "model.pkl")` and produces
platform-appropriate separators (`/` on Linux/Mac, `\` on Windows).

**`joblib.load(model_path)`** --- Deserializes the model from disk. This reconstructs the
exact scikit-learn estimator object that was saved during training, including:
- The learned coefficients or tree structures
- The hyperparameters
- The feature names (if trained on a DataFrame)
- The classes array (`[0, 1]` for our binary classifier)

**Why load both a model AND a scaler?** During training, we fit a `StandardScaler` on the
`Amount` column to normalize its distribution. That scaler learned the training set's mean
and standard deviation. At prediction time, we must apply the same transformation using the
same learned parameters. If we re-fit a scaler on prediction data, the normalization would
be different, and predictions would be wrong.

---

### `predict()` --- Single Prediction

```python
def predict(features: dict | pd.DataFrame) -> dict:
    if _model is None:
        load_artifacts()
```

**`features: dict | pd.DataFrame`** --- The function accepts two input types. This is
**duck typing in action** --- it does not matter what type you pass, as long as the function
can work with it. An API handler naturally has a dict (from JSON parsing). A batch pipeline
naturally has a DataFrame. Accepting both avoids forcing callers to convert.

**`if _model is None: load_artifacts()`** --- This is the lazy loading trigger. If nobody
called `load_artifacts()` explicitly before the first prediction, it happens automatically.
In production, the FastAPI lifespan calls `load_artifacts()` at startup so this branch is
never hit during request handling. But it serves as a safety net.

Why check `_model` and not `_scaler`? Because `load_artifacts()` always loads both. If
`_model` is not None, the scaler was loaded in the same call.

```python
    if isinstance(features, dict):
        features = pd.DataFrame([features])
```

**`isinstance(features, dict)`** --- Checks if the input is a dictionary. This is preferred
over `type(features) == dict` because `isinstance` also matches subclasses (like
`OrderedDict` or custom dict subclasses).

**`pd.DataFrame([features])`** --- Creates a single-row DataFrame from a dictionary. The list
wrapper `[features]` is critical. Here is why:

```python
# WITHOUT the list wrapper:
pd.DataFrame({"V1": 0.5, "V2": 1.0})
# ValueError: If using all scalar values, you must pass an index

# WITH the list wrapper:
pd.DataFrame([{"V1": 0.5, "V2": 1.0}])
#     V1   V2
# 0  0.5  1.0
```

When pandas sees a plain dict of scalars, it does not know how many rows you want. Wrapping
in a list says "this is a list of records, each dict is one row." The result is a DataFrame
with one row and columns matching the dict keys.

```python
    if "Amount" in features.columns and _scaler is not None:
        features = features.copy()
        features["Amount"] = _scaler.transform(features[["Amount"]])
```

**`features = features.copy()`** --- This is **defensive copying**. Without it, modifying
`features["Amount"]` would mutate the caller's original DataFrame. Here is the bug that
defensive copying prevents:

```python
# WITHOUT .copy():
my_data = pd.DataFrame([{"Amount": 100.0, "V1": 0.5}])
predict(my_data)
print(my_data["Amount"])  # Now shows -0.347 (scaled!) instead of 100.0
# The caller's data has been silently corrupted
```

Mutating input data is one of the most common bugs in production ML systems. It causes
cascading failures: a batch pipeline that calls `predict()` in a loop finds that data from
the first iteration has been modified by the time the second iteration runs.

**`_scaler.transform(features[["Amount"]])`** --- Note the double brackets: `[["Amount"]]`
not `["Amount"]`. Single brackets produce a pandas Series (1D), double brackets produce a
DataFrame (2D). Scikit-learn scalers expect 2D input because they are designed to handle
multiple columns at once. Passing a 1D Series would raise a `ValueError`.

**`.transform()` vs `.fit_transform()`** --- This is a crucial distinction:

- **`fit_transform()`** --- Used during training. First learns the parameters (mean, std) from
  the data, then applies the transformation. This was called when we built the scaler.
- **`transform()`** --- Used during prediction. Applies the already-learned parameters to new
  data. The mean and std were learned during training and saved inside the scaler object.

If you accidentally call `fit_transform()` at prediction time, the scaler would learn new
parameters from the incoming request data, which would be completely different from what the
model was trained with. Your predictions would be wrong, and every request would produce
different scaling parameters. This is a data leakage bug at inference time.

```python
    if "Time" in features.columns:
        features = features.drop(columns=["Time"])
```

The `Time` column in the credit card dataset represents seconds elapsed from the first
transaction. It was not used as a training feature (it would cause the model to learn
time-of-day patterns from one specific dataset rather than general fraud patterns). We drop
it here to ensure the feature set matches what the model expects.

**`features.drop(columns=["Time"])`** returns a new DataFrame with `Time` removed. The
original is not modified because `drop` returns a copy by default (unlike in-place operations).

```python
    assert _model is not None
    probability = float(_model.predict_proba(features)[:, 1][0])
    prediction = int(probability >= 0.5)
```

**`assert _model is not None`** --- A runtime guard that makes type checkers happy. After the
lazy loading check at the top, `_model` should never be None here. But mypy cannot reason
about global state across function calls, so the assert tells it "trust me, this is not None."
In production, if this assert fires, it means `load_artifacts()` failed silently, which would
be a critical bug.

**`_model.predict_proba(features)`** --- This is the core prediction call. For a binary
classifier, `predict_proba` returns a 2D numpy array with shape `(n_samples, 2)`:

```
[[0.92, 0.08],    # Sample 0: 92% class 0, 8% class 1
 [0.15, 0.85],    # Sample 1: 15% class 0, 85% class 1
 [0.50, 0.50]]    # Sample 2: 50-50
```

Column 0 contains the probability of class 0 (legitimate). Column 1 contains the probability
of class 1 (fraud). The two columns always sum to 1.0 for each row.

**`[:, 1]`** --- NumPy slicing. The colon `:` means "all rows," and `1` means "column index 1"
(the fraud probability). Result for a single prediction: `array([0.08])`.

**`[0]`** --- Extracts the first (and only) element from the 1D array: `0.08`.

**`float(...)`** --- Converts from `numpy.float64` to Python's native `float`. This matters
for JSON serialization. FastAPI uses Python's `json` module, which does not know how to
serialize numpy types. Without this conversion:

```python
import json
import numpy as np
json.dumps({"probability": np.float64(0.08)})
# TypeError: Object of type float64 is not JSON serializable
```

**`int(probability >= 0.5)`** --- Thresholding. `probability >= 0.5` returns a Python `bool`
(`True` or `False`). `int(True)` is `1`, `int(False)` is `0`. The threshold of 0.5 is the
default but can be tuned based on business requirements:

- Lower threshold (e.g., 0.3) = catch more fraud but more false positives
- Higher threshold (e.g., 0.7) = fewer false positives but some fraud slips through

In production systems like Stripe, thresholds are tuned per merchant, per card network, and
per region. A 0.5 threshold is a starting point, not a final answer.

```python
    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "label": "fraud" if prediction == 1 else "legitimate",
    }
```

The return dict provides three complementary views of the same prediction:

- **`prediction`** (int) --- machine-readable binary decision (0 or 1)
- **`fraud_probability`** (float) --- the raw confidence score, rounded to 6 decimal places
  to avoid floating-point noise like `0.0800000000000001`
- **`label`** (str) --- human-readable interpretation

Why all three? Different consumers need different things. A downstream rules engine wants the
int. A dashboard wants the probability. A notification to a fraud analyst wants the label.

---

### `predict_batch()` --- Batch Prediction Pattern

```python
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if _model is None:
        load_artifacts()

    features = df.copy()
    if "Amount" in features.columns and _scaler is not None:
        features["Amount"] = _scaler.transform(features[["Amount"]])
    if "Time" in features.columns:
        features = features.drop(columns=["Time"])

    assert _model is not None
    probabilities = _model.predict_proba(features)[:, 1]
    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = (probabilities >= 0.5).astype(int)
    return result
```

**Why a separate batch function instead of looping over `predict()`?** Performance.

Calling `predict()` in a loop processes one row at a time. Each call has overhead: DataFrame
creation, scaler transform setup, model prediction setup. For 100,000 transactions, that loop
might take 60 seconds.

`predict_batch()` passes the entire DataFrame to the model at once. Scikit-learn's
`predict_proba` is implemented in C/Cython and is optimized for batch operations. The same
100,000 transactions might complete in 2 seconds --- a 30x speedup.

**`probabilities = _model.predict_proba(features)[:, 1]`** --- Same slicing as before but
now returns a 1D array of 100,000 probabilities instead of a single scalar.

**`result = df.copy()`** --- Note we copy the original `df`, not the modified `features`. This
means the result DataFrame contains the original unscaled Amount and the Time column, plus
the new prediction columns. The caller gets their original data back with predictions appended.

**`(probabilities >= 0.5).astype(int)`** --- Vectorized thresholding. This is a numpy
operation that processes the entire array in one C-level loop:

```python
probabilities = np.array([0.08, 0.85, 0.50, 0.32])
probabilities >= 0.5
# array([False, True, True, False])

(probabilities >= 0.5).astype(int)
# array([0, 1, 1, 0])
```

The `.astype(int)` converts booleans to integers (False -> 0, True -> 1). This is vastly
faster than a Python list comprehension like `[1 if p >= 0.5 else 0 for p in probabilities]`
because it runs entirely in compiled numpy code with no Python interpreter overhead.

### Real-World Example: Stripe's Fraud Prediction at Scale

Stripe processes millions of transactions per day. Their fraud detection system, called Radar,
uses a pipeline strikingly similar to what we have built:

1. **Feature extraction** --- Each transaction is converted into hundreds of features (velocity,
   location, card history, merchant history). Our V1-V28 PCA components are an anonymized
   version of this.
2. **Real-time scoring** --- A model scores each transaction in under 100ms. This is our
   `predict()` function.
3. **Batch rescoring** --- Overnight jobs rescore historical transactions with updated models
   to catch patterns missed by the old model. This is our `predict_batch()`.
4. **Threshold tuning** --- Each merchant has configurable thresholds. A high-risk merchant
   might use 0.3, while a low-risk one uses 0.7. Our hardcoded 0.5 is the starting point.

The key architectural insight: the same prediction core (`predict` and `predict_batch`) serves
both real-time and batch use cases. The serving layer (FastAPI, Lambda) is just a wrapper.

---

## FastAPI Application (app.py) --- Line by Line

This file turns our prediction module into a production HTTP API. Every line here addresses a
real production concern: input validation, error handling, latency tracking, and monitoring.

### The Imports

```python
"""FastAPI application for fraud prediction serving."""

import time
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import load_artifacts, predict
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

**`time`** --- Python's built-in time module. Used to measure prediction latency. We use
`time.time()` rather than `time.perf_counter()` because we want wall-clock time in seconds
since the epoch, and the millisecond precision is sufficient for HTTP latency tracking.

**`asynccontextmanager`** from `contextlib` --- A decorator that turns an async generator
function into a context manager. This is used for FastAPI's lifespan protocol, which replaced
the older `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators (deprecated
in FastAPI 0.109+).

**`boto3`** --- The official AWS SDK for Python. It provides Python bindings for every AWS
service. Here we use it specifically for CloudWatch metrics publishing. The name "boto" comes
from the Amazon River dolphin (Boto), keeping with Amazon's river theme.

**`FastAPI`** --- The web framework class. FastAPI is built on:
- **Starlette** --- The ASGI web framework underneath
- **Pydantic** --- For request/response validation
- **Uvicorn** --- The ASGI server that actually handles TCP connections

How FastAPI compares to alternatives:

| Feature          | FastAPI              | Flask              | Django REST         |
|------------------|----------------------|--------------------|---------------------|
| Async support    | Native               | Bolt-on            | Bolt-on             |
| Validation       | Automatic (Pydantic) | Manual             | Serializers         |
| Auto docs        | Swagger + ReDoc      | None built-in      | Browsable API       |
| Performance      | ~15,000 req/s        | ~3,000 req/s       | ~2,500 req/s        |
| Type hints       | Required, used       | Optional, ignored  | Optional, ignored   |
| Learning curve   | Low                  | Low                | High                |

FastAPI is the dominant choice for ML serving because it combines high performance with
automatic request validation, which catches malformed inputs before they reach the model.

**`HTTPException`** --- FastAPI's way to return error responses with specific HTTP status codes.
Raising `HTTPException(status_code=500)` sends a `500 Internal Server Error` to the client.

**`BaseModel`** and **`Field`** from Pydantic --- Pydantic is a data validation library that
uses Python type annotations to define schemas. `BaseModel` is the base class for all
Pydantic models. `Field` provides additional constraints (min/max values, descriptions, etc.).

---

### The Lifespan: Startup and Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    logger.info("Model loaded --- ready to serve")
    yield
```

This function controls what happens when the application starts and stops.

**`@asynccontextmanager`** makes this function usable as an async context manager. The pattern
is: everything before `yield` runs at startup, and everything after `yield` runs at shutdown.

```python
# Conceptually equivalent to:
async def lifespan(app):
    # --- STARTUP ---
    load_artifacts()           # Load model from disk
    logger.info("...")         # Log readiness
    yield                      # Application runs and serves requests
    # --- SHUTDOWN ---
    # (nothing here, but we could close DB connections, flush buffers, etc.)
```

**Why load the model in lifespan and not at module level?**

Option 1 --- Module level (bad):
```python
# At the top of app.py
from src.models.predict import load_artifacts
load_artifacts()  # Runs when any file imports app.py
```
Problems: (a) importing app.py in tests triggers model loading, (b) model loading errors
happen at import time with confusing stack traces, (c) no way to pass custom paths.

Option 2 --- Lifespan (good):
```python
@asynccontextmanager
async def lifespan(app):
    load_artifacts()
    yield
```
Benefits: (a) model loads only when the server actually starts, (b) errors are clearly tied
to startup, (c) tests can import the app without loading models, (d) cleanup code goes after
yield.

The lifespan pattern replaced the older `@app.on_event("startup")` decorator because lifespan
supports resource cleanup and is compatible with ASGI lifecycle management.

---

### The FastAPI Application Instance

```python
app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Real-time credit card fraud detection",
    lifespan=lifespan,
)
```

Every parameter serves a purpose:

- **`title`** --- Appears in the auto-generated Swagger UI at `/docs`. When someone opens your
  API docs, this is the heading they see.
- **`version`** --- API version string. Displayed in docs and available programmatically. In a
  mature system, you might use this for API versioning (v1, v2).
- **`description`** --- Longer text explaining what the API does. Rendered in Swagger UI as
  markdown. This is documentation that lives with the code.
- **`lifespan`** --- The lifespan context manager we defined above. FastAPI calls it when the
  server starts and stops.

FastAPI automatically generates two documentation endpoints from this configuration:
- **`/docs`** --- Swagger UI (interactive, lets you test endpoints from the browser)
- **`/redoc`** --- ReDoc (read-only, cleaner for sharing with stakeholders)

---

### Pydantic Models: Input Validation

```python
class TransactionRequest(BaseModel):
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    # ... V3 through V27 follow the same pattern ...
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount")
```

**What is Pydantic?** Pydantic is a data validation library that uses Python type annotations
to validate data at runtime. When a JSON request comes into our API, Pydantic automatically:

1. Parses the JSON into Python types
2. Validates that every required field is present
3. Validates that types are correct (e.g., `V1` must be a float, not a string)
4. Validates constraints (e.g., `Amount >= 0`)
5. Returns a clean, typed object or a detailed error message

**`Field(..., description="PCA component V1")`** --- The three dots `...` (Python's `Ellipsis`
literal) mean "this field is required." There is no default value; omitting it causes a
validation error. Alternatives:

```python
V1: float = Field(..., description="...")       # Required, no default
V1: float = Field(default=0.0)                  # Optional, defaults to 0.0
V1: float = Field(default=None)                 # Optional, defaults to None
V1: float                                       # Required (implicit), no constraints
```

**`ge=0`** on the Amount field means "greater than or equal to zero." Pydantic supports:

| Constraint | Meaning                  | Example            |
|------------|--------------------------|--------------------|
| `ge`       | Greater than or equal    | `ge=0` (Amount>=0) |
| `gt`       | Strictly greater than    | `gt=0` (Amount>0)  |
| `le`       | Less than or equal       | `le=1.0`           |
| `lt`       | Strictly less than       | `lt=100`           |
| `min_length` | Min string length      | `min_length=3`     |
| `max_length` | Max string length      | `max_length=50`    |
| `regex`    | Pattern match            | `regex="^[A-Z]"`   |

**What happens when someone sends invalid data?**

```bash
# Missing required field V1:
curl -X POST /predict -d '{"V2": 1.0, "Amount": 50}'
# Response: 422 Unprocessable Entity
# {
#   "detail": [{
#     "type": "missing",
#     "loc": ["body", "V1"],
#     "msg": "Field required"
#   }]
# }

# Negative amount:
curl -X POST /predict -d '{"V1": 0.5, ..., "Amount": -10}'
# Response: 422 Unprocessable Entity
# {
#   "detail": [{
#     "type": "greater_than_equal",
#     "loc": ["body", "Amount"],
#     "msg": "Input should be greater than or equal to 0"
#   }]
# }

# String instead of float:
curl -X POST /predict -d '{"V1": "not_a_number", ...}'
# Response: 422 Unprocessable Entity
# {
#   "detail": [{
#     "type": "float_parsing",
#     "loc": ["body", "V1"],
#     "msg": "Input should be a valid number"
#   }]
# }
```

All of this validation happens before our `predict()` function is ever called. Bad data never
reaches the model. This is a massive safety net. Without it, a malformed request could cause
a numpy error deep inside the model, returning a confusing 500 error instead of a clear 422.

### Pydantic Models: Output Schema

```python
class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    label: str
    latency_ms: float
```

**Why define a response model?** Three reasons:

1. **Documentation** --- The Swagger UI shows exactly what the response looks like, including
   types and field names. Consumers of your API know exactly what to expect.

2. **Validation** --- If your `predict()` function returns unexpected data (e.g., a string
   where an int is expected), FastAPI catches it and returns a 500 error instead of sending
   malformed data to the client.

3. **Serialization** --- FastAPI uses the response model to filter the output. If `predict()`
   returns extra fields not in the response model, they are stripped out. This prevents
   accidentally leaking internal data.

**`latency_ms`** --- Note this field is not in the `predict()` return dict. It is added by
the endpoint handler after measuring how long the prediction took. This is a common pattern:
the core ML function returns pure predictions, and the serving layer adds operational metadata.

---

### The Health Check Endpoint

```python
@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}
```

**Why every production API needs a health check:**

1. **Load balancers** (AWS ALB, ELB, Kubernetes ingress) poll the health endpoint to decide
   whether to route traffic to this instance. If `/health` returns non-200, the instance is
   taken out of rotation.

2. **Container orchestrators** (ECS, Kubernetes) use health checks to decide whether to
   restart a container. If the health check fails 3 consecutive times, the container is killed
   and replaced.

3. **Monitoring systems** (DataDog, PagerDuty) alert on-call engineers when health checks fail.

4. **Deployment systems** use health checks for rolling deploys. A new version is only
   considered "deployed" when its health check passes.

**`@app.get("/health")`** --- This is a GET endpoint, not POST. Health checks are read-only
operations with no side effects, which is the semantic meaning of HTTP GET. Load balancers
send GET requests by default.

In a more sophisticated system, the health check would verify that the model is actually
loaded and functional:

```python
@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_type": type(_model).__name__}
```

---

### The Prediction Endpoint

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

**`@app.post("/predict", response_model=PredictionResponse)`** --- Why POST and not GET?

- **GET** is for retrieving data. It should have no side effects, and parameters go in the URL
  query string. Our transaction has 29 float fields --- putting those in a URL would be ugly
  and hit URL length limits.
- **POST** is for submitting data to be processed. The request body is JSON, which cleanly
  holds our 29 fields. Additionally, POST requests are not cached by browsers or CDNs, which
  is correct because each prediction is unique.

**`response_model=PredictionResponse`** tells FastAPI to validate the response against
`PredictionResponse` and to show the response schema in the auto-generated docs.

**`transaction: TransactionRequest`** --- FastAPI sees this type annotation and automatically:
1. Reads the request body as JSON
2. Validates it against `TransactionRequest`
3. Returns 422 if validation fails
4. Passes the validated Pydantic object to the function

**`start = time.time()`** --- Captures the current time in seconds since epoch (e.g.,
`1700000000.123456`). Placing it before the try block means we start the timer even if the
prediction fails, which is important for error latency tracking.

**`transaction.model_dump()`** --- Pydantic v2 method that converts the Pydantic model to a
plain Python dictionary. In Pydantic v1, this was `.dict()`, but `.dict()` is deprecated.

```python
transaction = TransactionRequest(V1=0.5, V2=1.0, ..., Amount=100.0)
transaction.model_dump()
# {"V1": 0.5, "V2": 1.0, ..., "Amount": 100.0}
```

We convert to dict because `predict()` accepts `dict | pd.DataFrame`. The Pydantic model
itself is not a dict, so we need this conversion.

**`latency = (time.time() - start) * 1000`** --- Calculates elapsed time. Subtracting start
from current time gives seconds, multiplying by 1000 converts to milliseconds. A healthy
prediction should complete in 1-50ms. If this number creeps above 100ms, something is wrong
(model too large, feature preprocessing bottleneck, etc.).

**Why track prediction latency?** Because latency directly impacts user experience and
business metrics. In Stripe's fraud system, every millisecond of latency in fraud scoring
adds perceived latency to checkout. At scale, p99 latency matters more than average:

- **Average latency: 15ms** (looks fine)
- **p99 latency: 800ms** (1 in 100 users waits almost a second --- not fine)

**`_publish_metrics(result)`** --- Publishes prediction metrics to CloudWatch. Note this is
called after the prediction succeeds but before the response is returned. This adds a small
amount of latency (~10-50ms for the CloudWatch API call). In a high-performance system, you
would publish metrics asynchronously (via a background queue) to avoid blocking the response.

**`except Exception as e:`** --- Catches any exception from the prediction pipeline. This is
a broad catch that handles:
- Model loading failures (file not found)
- Feature shape mismatches (wrong number of columns)
- Numpy/pandas errors (NaN values, type mismatches)
- Any other unexpected error

**`logger.error(f"Prediction failed: {e}")`** --- Logs the error with full context. In
production, these logs go to CloudWatch Logs, where alerts can be configured to page on-call
engineers if error rate exceeds a threshold.

**`raise HTTPException(status_code=500, detail=str(e))`** --- Converts the Python exception
into an HTTP 500 response. The `detail` field is included in the JSON response body so the
caller knows what went wrong. In a security-sensitive production system, you might sanitize
the error message to avoid leaking internal details:

```python
raise HTTPException(status_code=500, detail="Internal prediction error")
```

---

### CloudWatch Metrics Publishing

```python
def _publish_metrics(result: dict):
    try:
        cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
        cloudwatch.put_metric_data(
            Namespace="MLOps/FraudDetection",
            MetricData=[
                {
                    "MetricName": "PredictionLatency",
                    "Value": result["latency_ms"],
                    "Unit": "Milliseconds",
                },
                {
                    "MetricName": "FraudPredicted",
                    "Value": result["prediction"],
                    "Unit": "Count",
                },
                {
                    "MetricName": "FraudProbability",
                    "Value": result["fraud_probability"],
                    "Unit": "None",
                },
            ],
        )
    except Exception as e:
        logger.warning(f"Failed to publish CloudWatch metrics: {e}")
```

**`boto3.client("cloudwatch", region_name="us-east-1")`** --- Creates a CloudWatch client.
`boto3.client()` creates a low-level service client. The `region_name` specifies which AWS
region to publish metrics to. In production, this would typically come from environment
variables or instance metadata rather than being hardcoded.

Note: creating a new boto3 client on every request is suboptimal. Each `boto3.client()` call
involves connection setup. In a high-throughput system, you would create the client once at
module level:

```python
# Better: create once, reuse
_cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def _publish_metrics(result: dict):
    _cloudwatch.put_metric_data(...)
```

**`Namespace="MLOps/FraudDetection"`** --- CloudWatch namespaces are like folders for metrics.
AWS services use namespaces like `AWS/EC2` and `AWS/Lambda`. Custom namespaces must not start
with `AWS/`. Our namespace groups all fraud detection metrics together.

**`MetricData`** --- A list of metric data points. Each entry has:

- **`MetricName`** --- The metric name. Choose names that are clear in a dashboard:
  `PredictionLatency`, not `latency` or `metric_1`.
- **`Value`** --- The numeric value. For latency, this is milliseconds. For FraudPredicted,
  it is 0 or 1. For probability, it is a float between 0 and 1.
- **`Unit`** --- The unit of measurement. CloudWatch uses this for aggregation (you can
  compute average latency in milliseconds, total count of fraud predictions, etc.). `"None"`
  is a special unit for dimensionless values like probabilities.

**Why wrap the entire function in try/except?** Because metric publishing should never break
predictions. If CloudWatch is down or credentials are expired, the prediction should still
return successfully. The metrics are operational telemetry, not business logic. Failing to
publish a metric is a warning, not an error.

This is the **bulkhead pattern** in action: the failure of one non-critical component
(metrics) is isolated from the critical path (predictions).

### Real-World Example: Building a 10,000 RPS Fraud API

At 10,000 requests per second, every design decision matters. Here is how a production-grade
system would evolve from our codebase:

1. **Connection pooling** --- Move `boto3.client()` to module level. Create the client once,
   reuse it. Saves ~50ms per request in connection overhead.

2. **Async metrics** --- Publish metrics to a local queue (Redis, SQS, or an in-memory buffer)
   and let a background thread flush them to CloudWatch. This removes CloudWatch latency from
   the critical path entirely.

3. **Model caching** --- Use `functools.lru_cache` or a dedicated model registry to avoid
   reloading models. Version models with hashes so cache invalidation is automatic.

4. **Horizontal scaling** --- Run multiple instances behind a load balancer. Each instance
   handles ~2,000 RPS, so 5 instances cover 10,000 RPS with headroom.

5. **Batching at the edge** --- For burst traffic, buffer incoming requests and call
   `predict_batch()` every 10ms. This trades a small latency increase for much higher
   throughput.

6. **Feature store** --- Replace raw feature passing with a feature store lookup (Feast, Tecton,
   or a custom Redis-based store). The API receives a transaction ID, looks up precomputed
   features, and passes those to the model.

---

## Lambda Handler (lambda_handler.py) --- Line by Line

This is the shortest file in the project but connects two major worlds: serverless computing
and web frameworks.

### The Full File

```python
"""AWS Lambda handler wrapping the FastAPI app via Mangum."""

from mangum import Mangum

from src.serving.app import app

handler = Mangum(app, lifespan="on")
```

That is the entire file --- three meaningful lines. But each line carries significant weight.

---

### What is AWS Lambda?

AWS Lambda is a **serverless compute service**. You upload your code, and AWS runs it in
response to events (HTTP requests, S3 uploads, SQS messages, scheduled triggers). You do not
manage servers, operating systems, or scaling. AWS handles all of that.

Key characteristics:

- **No servers to manage** --- AWS provisions, patches, and scales the infrastructure.
- **Pay per request** --- You are billed for the number of invocations and the duration of
  each invocation, measured in milliseconds. No requests = no cost.
- **Auto-scaling** --- Lambda scales from 0 to thousands of concurrent executions automatically.
  If traffic spikes from 10 to 10,000 requests/second, Lambda spins up new execution
  environments without intervention.
- **Ephemeral** --- Each Lambda execution runs in a sandboxed environment. The environment may
  be reused for subsequent invocations (warm start) or destroyed (cold start).
- **Resource limits** --- Maximum 15 minutes per invocation, 10GB memory, 10GB ephemeral
  storage.

For ML serving, Lambda is attractive for workloads with variable traffic. A fraud detection
API might handle 100 requests/second during business hours and 5 requests/second at 3 AM.
With Lambda, you pay only for what you use. With a traditional server (EC2, ECS), you pay for
the instance even when it is idle.

---

### What is Mangum?

**Mangum** is an adapter library that translates between AWS Lambda's event format and the
ASGI protocol that FastAPI speaks. The name is a play on "mango" and has no deeper meaning.

The problem Mangum solves:

```
AWS Lambda receives:
{
    "httpMethod": "POST",
    "path": "/predict",
    "body": "{\"V1\": 0.5, ...}",
    "headers": {"Content-Type": "application/json"}
}

FastAPI expects (ASGI):
{
    "type": "http.request",
    "method": "POST",
    "path": "/predict",
    "body": b'{"V1": 0.5, ...}',
    "headers": [(b"content-type", b"application/json")]
}
```

These are completely different formats. Mangum translates between them. Without Mangum, you
would need to write this translation layer yourself, handling edge cases like multipart
uploads, WebSocket connections, and binary responses.

---

### Line-by-Line Breakdown

```python
from mangum import Mangum
```

Imports the Mangum adapter class. Mangum is a lightweight library (~500 lines of code) with
no heavy dependencies.

```python
from src.serving.app import app
```

Imports our FastAPI application instance. This import also triggers the import of all the
endpoint definitions, Pydantic models, and the lifespan function. But because of our lazy
loading design, it does NOT load the ML model yet.

```python
handler = Mangum(app, lifespan="on")
```

**`handler`** --- This variable name is significant. AWS Lambda looks for a callable named
`handler` (configurable in Lambda settings, but `handler` is the convention). When a request
arrives, Lambda calls `handler(event, context)` where `event` is the API Gateway request and
`context` contains Lambda metadata (remaining time, function name, etc.).

**`Mangum(app, lifespan="on")`** --- Creates the Mangum adapter wrapping our FastAPI app.

**`lifespan="on"`** --- This parameter controls how Mangum handles FastAPI's lifespan events.
The options are:

| Value   | Behavior                                                              |
|---------|-----------------------------------------------------------------------|
| `"on"`  | Lifespan startup runs on first invocation, shutdown runs on termination |
| `"off"` | Lifespan events are ignored entirely                                  |
| `"auto"`| Mangum tries to detect if lifespan is supported                       |

We use `"on"` because our lifespan function loads the ML model. With `"off"`, the model would
never be loaded, and the first prediction call would trigger lazy loading inside the Lambda
request handler (adding model loading latency to the first request).

With `"on"`, the flow is:

1. Lambda cold start: Python initializes, modules are imported
2. First request arrives: Mangum triggers the lifespan startup event
3. `load_artifacts()` runs, loading model and scaler into memory
4. The request is processed
5. Subsequent requests skip model loading (it is already in memory)
6. When Lambda recycles the execution environment, the lifespan shutdown event fires

---

### How a Request Flows Through the Stack

Here is the complete request lifecycle, from a user's HTTP call to the prediction response:

```
Client (curl/browser/app)
    |
    | HTTPS POST /predict {"V1": 0.5, ..., "Amount": 100}
    v
API Gateway
    |
    | Transforms HTTP request into Lambda event format
    | Adds request ID, API key validation, rate limiting
    v
AWS Lambda
    |
    | Invokes handler(event, context)
    v
Mangum
    |
    | Translates Lambda event -> ASGI scope + receive
    | Calls app(scope, receive, send)
    v
FastAPI (Starlette)
    |
    | Routes /predict to predict_fraud()
    | Parses JSON body into TransactionRequest (Pydantic)
    | Returns 422 if validation fails
    v
predict_fraud()
    |
    | Calls predict(transaction.model_dump())
    | Measures latency
    | Publishes CloudWatch metrics
    v
predict()
    |
    | Scales Amount, drops Time
    | Calls model.predict_proba()
    | Returns prediction dict
    v
FastAPI (Starlette)
    |
    | Validates response against PredictionResponse
    | Serializes to JSON
    v
Mangum
    |
    | Translates ASGI response -> Lambda response format
    v
API Gateway
    |
    | Transforms Lambda response into HTTP response
    | Adds CORS headers, etc.
    v
Client receives:
{
    "prediction": 1,
    "fraud_probability": 0.856432,
    "label": "fraud",
    "latency_ms": 12.45
}
```

Each layer adds value: API Gateway handles authentication and rate limiting, Mangum handles
protocol translation, FastAPI handles validation and routing, and our predict module handles
the actual ML inference.

---

### Cold Starts: What They Are, Why They Happen, How to Mitigate

A **cold start** occurs when Lambda must create a new execution environment from scratch. This
involves:

1. **Provisioning** --- AWS allocates a micro-VM (Firecracker) for your function (~100ms)
2. **Downloading code** --- Your deployment package (container image or zip) is downloaded
   from S3/ECR (~100-2000ms depending on package size)
3. **Runtime initialization** --- Python interpreter starts, your code is imported (~200-500ms)
4. **Application initialization** --- Lifespan runs, model loads from disk (~500-2000ms)

Total cold start for our fraud detection API: roughly 1-4 seconds, depending on model size
and container image size.

**When do cold starts happen?**

- After a period of inactivity (Lambda recycles idle environments after ~5-15 minutes)
- When traffic spikes and Lambda needs more concurrent environments
- After a code deployment (all environments are replaced)
- When Lambda proactively recycles environments (for security/maintenance)

**How to mitigate cold starts:**

1. **Provisioned Concurrency** --- AWS keeps N environments warm at all times. You pay for
   idle compute, but cold starts are eliminated. Configure in Lambda settings:
   ```bash
   aws lambda put-provisioned-concurrency-config \
       --function-name fraud-detection \
       --qualifier prod \
       --provisioned-concurrent-executions 10
   ```
   This keeps 10 warm environments ready. The first 10 concurrent requests never hit a cold
   start.

2. **Smaller container images** --- Our Docker image should be as lean as possible. Use slim
   base images, remove unnecessary files, use multi-stage builds. A 500MB image downloads
   in ~2s; a 2GB image takes ~8s.

3. **Lazy imports** --- Import heavy libraries inside functions rather than at module level:
   ```python
   # Module-level (adds to cold start):
   import tensorflow as tf

   # Lazy (deferred until first use):
   def predict():
       import tensorflow as tf
       ...
   ```
   This helps when not all requests need all libraries.

4. **SnapStart** (Java/Python preview) --- AWS snapshots the initialized environment and
   restores it on cold start, skipping initialization entirely. Currently available for Java,
   with Python support in development.

5. **Keep-warm pings** --- A CloudWatch Events rule that invokes the Lambda every 5 minutes
   to prevent the environment from being recycled:
   ```json
   {
       "schedule": "rate(5 minutes)",
       "target": "arn:aws:lambda:...:fraud-detection"
   }
   ```
   Cheap but unreliable --- only keeps one environment warm. If you need 10 concurrent
   environments, you need provisioned concurrency.

### Real-World Example: Lambda Cold Start Adding 3 Seconds

A fintech company deployed their fraud model on Lambda with a 1.5GB container image. During
off-peak hours, Lambda recycled all execution environments. The first transaction of the
morning consistently timed out because:

- Container image download: 3 seconds
- Python startup + imports: 1 second
- Model loading from S3: 2 seconds
- Total cold start: 6 seconds
- API Gateway timeout: 5 seconds (default)

The fix was a three-pronged approach:

1. **Reduced image size from 1.5GB to 400MB** by switching from `python:3.11` to
   `python:3.11-slim`, removing test dependencies, and using multi-stage Docker builds.
   Cold start dropped from 6s to 3s.

2. **Enabled provisioned concurrency with 5 instances** during business hours (8 AM - 10 PM).
   This eliminated cold starts for the first 5 concurrent requests. Monthly cost: ~$50.

3. **Increased API Gateway timeout to 29 seconds** (the maximum) as a safety net. This
   prevented timeouts even if a cold start did occur.

Result: p99 latency dropped from 6 seconds to 45 milliseconds. The 5 provisioned instances
handled normal traffic, and auto-scaling covered spikes.

---

## API Design Patterns for ML

This section covers patterns that are not in our code but that you will be asked about in
interviews and will need in production systems.

### 1. Request/Response Pattern

This is what our API implements: a client sends a request, waits for a response.

```
Client ---POST /predict---> Server ---response---> Client
         (synchronous, blocking)
```

**When to use:** When the client needs the prediction immediately (fraud detection during
checkout, content moderation during upload, autocomplete during typing).

**Latency budget:** The total time from request to response must fit within the client's
tolerance. For fraud detection at checkout, this is typically 100-500ms. For content
moderation, it might be 1-2 seconds.

**Our implementation** follows this pattern perfectly. The client sends a POST, the server
runs `predict()`, and returns the result. Simple, predictable, debuggable.

### 2. Synchronous vs Asynchronous Prediction

**Synchronous** (our approach):
```
Client --> POST /predict --> wait --> response
           Time: 15ms
```

**Asynchronous** (for expensive predictions):
```
Client --> POST /predict --> 202 Accepted (with job_id)
Client --> GET /predict/job_123 --> 200 (result) or 202 (still processing)
```

The async pattern is used when prediction takes too long for a synchronous call. Examples:

- **Image generation** (DALL-E): 10-60 seconds per image. The API returns a job ID, and the
  client polls or receives a webhook when the image is ready.
- **Video analysis**: Processing a 2-minute video through a vision model takes 30+ seconds.
- **Ensemble predictions**: Running 10 models and aggregating results might take 5 seconds.

Implementation sketch:

```python
from uuid import uuid4
jobs = {}  # In production, use Redis or DynamoDB

@app.post("/predict/async")
async def predict_async(transaction: TransactionRequest):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "processing"}
    # In production: submit to SQS/Celery/background task
    background_tasks.add_task(run_prediction, job_id, transaction)
    return {"job_id": job_id, "status": "processing"}

@app.get("/predict/status/{job_id}")
async def get_prediction(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
```

### 3. Batch Endpoint Pattern

Accepts multiple predictions in one request. More efficient than making N individual requests
because it amortizes HTTP overhead and allows the model to process inputs as a batch.

```python
class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_latency_ms: float

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch_endpoint(batch: BatchRequest):
    start = time.time()
    df = pd.DataFrame([t.model_dump() for t in batch.transactions])
    results = predict_batch(df)
    latency = (time.time() - start) * 1000
    predictions = [
        PredictionResponse(
            prediction=row["prediction"],
            fraud_probability=row["fraud_probability"],
            label="fraud" if row["prediction"] == 1 else "legitimate",
            latency_ms=latency / len(batch.transactions),
        )
        for _, row in results.iterrows()
    ]
    return BatchResponse(predictions=predictions, total_latency_ms=round(latency, 2))
```

**When to use:** When clients naturally have batches of data (end-of-day transaction scoring,
marketing campaign audience scoring) or when you want to maximize throughput.

**Sizing considerations:** A batch of 1,000 transactions might take 200ms, while 1,000
individual requests would take 15,000ms (15ms each). But very large batches can cause
timeouts (Lambda's 15-minute limit, API Gateway's 29-second limit). Cap batch size at a
reasonable number:

```python
class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=1000)
```

### 4. Health Check and Readiness Probes

Kubernetes (and ECS) distinguish between two types of health checks:

**Liveness probe** --- "Is the process alive?" If this fails, kill the container and restart.
```python
@app.get("/health/live")
def liveness():
    return {"status": "alive"}
```

**Readiness probe** --- "Is the process ready to accept traffic?" If this fails, stop sending
traffic but do not kill the container (it might be loading the model).
```python
@app.get("/health/ready")
def readiness():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Optionally run a test prediction to verify the model works
    try:
        test_input = {f"V{i}": 0.0 for i in range(1, 29)}
        test_input["Amount"] = 0.0
        predict(test_input)
        return {"status": "ready", "model": "verified"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model broken: {e}")
```

**Our `/health` endpoint** is a simple liveness check. In a production Kubernetes deployment,
you would split it into `/health/live` and `/health/ready`.

Why the distinction matters: during a rolling deployment, the new container is starting up
and loading the model. Liveness: alive (process is running). Readiness: not ready (model still
loading). Kubernetes keeps routing traffic to the old containers until the new one becomes
ready. Without this distinction, traffic would be routed to the new container before the
model is loaded, causing 503 errors.

### 5. API Versioning for ML Models

ML models change frequently. You might deploy a new model version every week. API versioning
ensures old clients continue to work while new clients use the latest model.

**URL path versioning** (most common):
```
POST /v1/predict   --> uses model v1 (Random Forest)
POST /v2/predict   --> uses model v2 (XGBoost)
```

**Header versioning:**
```
POST /predict
X-Model-Version: 2
```

**Query parameter versioning:**
```
POST /predict?model_version=2
```

Implementation with FastAPI routers:

```python
from fastapi import APIRouter

v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

@v1_router.post("/predict")
def predict_v1(transaction: TransactionRequest):
    return predict(transaction.model_dump(), model_version="v1")

@v2_router.post("/predict")
def predict_v2(transaction: TransactionRequestV2):  # might have different features
    return predict(transaction.model_dump(), model_version="v2")

app.include_router(v1_router)
app.include_router(v2_router)
```

**When to version:** When the new model has different input features, different output format,
or significantly different behavior (different thresholds, different feature engineering).

**When NOT to version:** When you are simply retraining the same architecture on newer data.
The API contract has not changed, so no versioning is needed. Just swap the model file.

### 6. Rate Limiting and Throttling

Protecting your ML API from overload is critical. A single client sending 100,000 requests
per second can exhaust model serving capacity for everyone.

**API Gateway level** (preferred for AWS):
```yaml
# serverless.yml or CloudFormation
ApiGateway:
  throttle:
    burstLimit: 500      # Max concurrent requests
    rateLimit: 1000      # Requests per second
  quota:
    limit: 1000000       # Monthly request limit
    period: MONTH
```

**Application level** (using slowapi, a rate limiter for FastAPI):
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
def predict_fraud(request: Request, transaction: TransactionRequest):
    ...
```

**Why rate limit ML APIs specifically?** ML predictions are compute-intensive. A text endpoint
might handle 50,000 requests/second on a single server. An ML prediction endpoint might handle
2,000 requests/second. Without rate limiting, a traffic spike can cause cascading failures:
requests queue up, memory usage climbs, garbage collection pauses lengthen, and eventually
the server OOMs and crashes.

### 7. Serving Multiple Model Versions Simultaneously

A real-world pattern for A/B testing and gradual rollouts.

```
                          +-- Model v1 (90% traffic) --+
                          |                             |
Client --> API Gateway --+|                             |--> Response
                          |                             |
                          +-- Model v2 (10% traffic) --+
```

Implementation:

```python
import random

models = {
    "v1": load_model("models/v1/model.pkl"),
    "v2": load_model("models/v2/model.pkl"),
}

traffic_split = {"v1": 0.9, "v2": 0.1}

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    # Weighted random selection
    version = random.choices(
        list(traffic_split.keys()),
        weights=list(traffic_split.values()),
    )[0]

    model = models[version]
    result = predict_with_model(model, transaction.model_dump())
    result["model_version"] = version

    # Log for analysis: compare v1 vs v2 performance
    logger.info(f"Prediction: version={version}, fraud_prob={result['fraud_probability']}")

    return result
```

This pattern enables:

- **A/B testing** --- Compare model v2's fraud detection rate against v1 on live traffic
- **Canary deployment** --- Start with 1% traffic to v2, monitor for errors, gradually increase
- **Shadow mode** --- Run both models on every request, return v1's response, log v2's
  prediction for offline comparison (no risk to users)

Real-world example: a payments company wants to deploy a new XGBoost model to replace their
Random Forest. They configure 95% of traffic to the existing model and 5% to the new one. After
one week, they compare:

- Fraud catch rate: v2 catches 12% more fraud
- False positive rate: v2 has 8% fewer false positives
- Latency: v2 is 20ms faster (XGBoost is faster than Random Forest at inference)

Confident in the results, they gradually shift traffic: 5% -> 25% -> 50% -> 100%, monitoring
each step. The old model stays loaded as a rollback target for another week before being
removed.

---

## Interview Questions

### Conceptual Questions

**Q: Why not reload the model on every request?**
A: Model loading involves disk I/O and deserialization (200-500ms). With 1,000 requests/second,
that would be 200-500 seconds of compute per second --- physically impossible. Module-level
global variables persist for the process lifetime, so the model loads once and serves millions
of requests.

**Q: What is the difference between `predict_proba` and `predict`?**
A: `predict()` returns the class label (0 or 1). `predict_proba()` returns the probability
for each class. We use `predict_proba` because:
1. It gives us confidence scores, not just binary decisions
2. We can tune the threshold post-deployment without retraining
3. Downstream systems can make nuanced decisions (block high-probability fraud, flag medium
   probability for review)

**Q: Why use Pydantic models instead of just parsing raw JSON?**
A: Pydantic provides (1) type validation (floats are actually floats), (2) constraint
validation (Amount >= 0), (3) automatic API documentation, (4) clear error messages for
invalid requests. Without it, a string in a float field would cause a cryptic numpy error
deep in the model instead of a clean 422 response.

**Q: How do you handle model versioning in production?**
A: Multiple strategies: (1) URL path versioning `/v1/predict`, (2) header-based
`X-Model-Version: 2`, (3) traffic splitting for A/B testing, (4) shadow mode for risk-free
comparison. The choice depends on whether the API contract changes and whether you need
gradual rollouts.

**Q: What is a cold start and how do you mitigate it?**
A: A cold start occurs when Lambda must initialize a new execution environment (provision VM,
download code, start runtime, load model). Mitigation: provisioned concurrency (keeps
environments warm), smaller images (faster download), lazy imports (faster initialization),
and keep-warm pings (prevent recycling).

### Design Questions

**Q: Design a fraud detection API that handles 50,000 requests per second.**
A: At 50K RPS, Lambda alone will not work (concurrency limits, cost). Use:
1. ECS/Kubernetes cluster with autoscaling (20 instances, each handling 2,500 RPS)
2. Application load balancer with health-check routing
3. Model loaded once at container startup (lifespan pattern)
4. Async metrics publishing (buffer in memory, flush every 5 seconds)
5. Feature store (Redis) to avoid passing raw features over HTTP
6. Model versioning with traffic splitting for safe deployments
7. Circuit breaker pattern: if model errors exceed 5%, fall back to a rules-based system

**Q: How would you add A/B testing to this API?**
A: Load both model versions at startup. On each request, hash the user/transaction ID to
deterministically assign a model version (better than random because the same user always
gets the same version). Log the model version with every prediction. After sufficient data,
compare metrics (precision, recall, false positive rate) between versions.

**Q: How would you handle a model that takes 30 seconds to run?**
A: Switch from synchronous to asynchronous prediction:
1. POST /predict returns 202 Accepted with a job_id
2. Prediction runs in background (SQS + Lambda, Celery, or ECS task)
3. Client polls GET /predict/status/{job_id} or receives a webhook
4. Consider batching multiple requests to amortize model loading

### Debugging Questions

**Q: A prediction endpoint returns correct results locally but wrong results in production. How do you debug?**
A:
1. Check model version --- is the same model.pkl deployed? (Compare file hashes)
2. Check scaler version --- same scaler.pkl? (A mismatched scaler produces silently wrong results)
3. Check feature order --- DataFrames created from dicts have arbitrary column order in older pandas versions
4. Check input preprocessing --- is Amount being scaled? Is Time being dropped?
5. Check data types --- float32 vs float64 can cause tiny differences that compound
6. Log the raw features and scaled features at each stage
7. Send the same input to both environments and diff the intermediate values

**Q: Your API's p99 latency suddenly jumped from 20ms to 500ms. What do you investigate?**
A:
1. CloudWatch metrics --- is it all requests or a subset?
2. Model size --- did a new deployment include a larger model?
3. Memory pressure --- is the Lambda/container swapping to disk?
4. Garbage collection --- large numpy arrays can trigger expensive GC pauses
5. Cold starts --- did traffic spike cause many concurrent cold starts?
6. Downstream dependencies --- is CloudWatch metric publishing suddenly slow?
7. Data shape --- are inputs suddenly larger (more features, unexpected columns)?
