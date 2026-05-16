# Chapter 10: Data Pipeline Deep Dive -- Line-by-Line Code Explanation

This guide dissects every line of the data pipeline: configuration loading, logging,
data ingestion, validation, and preprocessing. Every function, every parameter, every
design decision is explained with real-world context. By the end, you will be able to
explain each piece in an interview and know why it was built this way.

---

## Table of Contents

1. [Configuration Management (config.py)](#1-configuration-management-configpy)
2. [Logging (logger.py)](#2-logging-loggerpy)
3. [Data Ingestion (ingest.py)](#3-data-ingestion-ingestpy)
4. [Data Validation (validate.py)](#4-data-validation-validatepy)
5. [Data Preprocessing (preprocess.py)](#5-data-preprocessing-preprocesspy)

---

## 1. Configuration Management (config.py)

**File:** `src/utils/config.py`

Configuration management is the spine of any MLOps project. It is what separates a
reproducible, auditable pipeline from a mess of hardcoded paths scattered across a
dozen scripts. Let us walk through every line.

### Full Source Code

```python
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_params(path: str | None = None) -> dict[str, Any]:
    if path is None:
        path = str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return dict(yaml.safe_load(f))


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_aws_config(params: dict | None = None) -> dict:
    if params is None:
        params = load_params()
    return {
        "region": os.getenv("AWS_REGION", params["aws"]["region"]),
        "s3_bucket": os.getenv("S3_BUCKET", params["aws"]["s3_bucket"]),
        "ecr_repository": params["aws"]["ecr_repository"],
        "lambda_function": params["aws"]["lambda_function"],
    }
```

### Line-by-Line Breakdown

#### Imports

```python
import os
from pathlib import Path
from typing import Any
import yaml
```

- **`import os`** -- The `os` module provides portable access to operating system
  functionality. Here it is used for `os.getenv()` to read environment variables.
  This is how your code adapts to the environment it runs in (your laptop vs a CI
  runner vs a production server) without changing a single line of code.

- **`from pathlib import Path`** -- `pathlib.Path` is the modern Python way to handle
  filesystem paths. Before `pathlib` (introduced in Python 3.4), you would write
  something like `os.path.join(os.path.dirname(__file__), '..', '..', 'params.yaml')`.
  `Path` objects support the `/` operator for joining, have methods like `.parent`,
  `.resolve()`, `.mkdir()`, and are cross-platform (they handle Windows backslashes
  vs Unix forward slashes automatically).

- **`from typing import Any`** -- `Any` is a type hint that means "this could be any
  type." It is used in the return type `dict[str, Any]` because a YAML file can
  contain strings, ints, floats, nested dicts, lists -- anything.

- **`import yaml`** -- The PyYAML library. YAML (YAML Ain't Markup Language) is the
  standard configuration format in MLOps because it is human-readable, supports
  comments (unlike JSON), and handles nested structures cleanly. You will see it in
  DVC, MLflow, Kubernetes, Docker Compose, GitHub Actions, and virtually every ML
  tool.

#### PROJECT_ROOT -- The Directory Traversal

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
```

This single line is doing a lot. Let us unpack it step by step.

**`__file__`** is a special Python variable that holds the path to the current script.
When Python runs `src/utils/config.py`, `__file__` contains something like
`/Users/manishsingh/Desktop/miss/miss/mlops/src/utils/config.py`.

**`Path(__file__)`** wraps that string in a `Path` object so we can use path
manipulation methods.

**`.resolve()`** converts the path to an absolute path and resolves any symlinks. If
`__file__` were a relative path like `src/utils/config.py`, `.resolve()` would turn
it into the full absolute path. This is critical because relative paths change
depending on where you run the script from.

**`.parent.parent.parent`** walks up the directory tree three times:

```
/Users/manishsingh/Desktop/miss/miss/mlops/src/utils/config.py   <-- __file__
/Users/manishsingh/Desktop/miss/miss/mlops/src/utils/            <-- .parent (1st)
/Users/manishsingh/Desktop/miss/miss/mlops/src/                  <-- .parent (2nd)
/Users/manishsingh/Desktop/miss/miss/mlops/                      <-- .parent (3rd) = PROJECT_ROOT
```

**Why this pattern?** Because `config.py` lives at `src/utils/config.py`, which is
three levels deep from the project root. The project root is where `params.yaml`,
`dvc.yaml`, and the `data/` directory live. By computing the root relative to the
file's own location, the code works regardless of:
- What directory you `cd` into before running it
- Whether you run it directly, via pytest, via DVC, or via a Docker container
- Whether the absolute path changes (e.g., different machines)

**Why is this a module-level constant?** It is computed once when the module is first
imported, then reused. This is efficient and guarantees consistency -- every function
in the project sees the same root path.

**Interview tip:** This is a very common pattern. An alternative is to use a
`.env` file or `python-dotenv`, but computing the root from `__file__` is simpler and
has zero dependencies.

#### load_params() -- Loading the YAML Configuration

```python
def load_params(path: str | None = None) -> dict[str, Any]:
    if path is None:
        path = str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return dict(yaml.safe_load(f))
```

**`path: str | None = None`** -- The parameter accepts either a string path or
`None`. The `str | None` syntax (union type with `|`) requires Python 3.10+. In
older Python you would write `Optional[str]`. When `None` is passed (the default),
the function falls back to the standard location.

**Why allow a custom path?** Testing. In unit tests, you might want to load a
different `params.yaml` with test-specific values (smaller dataset, fewer epochs).
In integration tests, you might point to a staging configuration. This is the
**Dependency Injection** pattern -- instead of hardcoding a path, you accept it as
a parameter so the caller can override it.

**`path = str(PROJECT_ROOT / "params.yaml")`** -- The `/` operator on a `Path` object
joins path segments. `PROJECT_ROOT / "params.yaml"` produces something like
`Path('/Users/.../mlops/params.yaml')`. We wrap it in `str()` because `open()` in
older Python versions does not accept `Path` objects (though modern Python does).

**`with open(path) as f:`** -- The `with` statement is a **context manager**. It
guarantees the file is closed when the block exits, even if an exception occurs.
Without `with`, you would need a try/finally block. File handles are a limited
resource -- if you forget to close them, you can hit the OS limit and get
"Too many open files" errors in production.

**`yaml.safe_load(f)`** -- This is where the YAML content becomes a Python dictionary.

**CRITICAL: `yaml.safe_load` vs `yaml.load`**

`yaml.load` (without `Loader` argument) is **dangerous**. YAML has a feature called
"tags" that can instantiate arbitrary Python objects. A malicious YAML file could
contain:

```yaml
exploit: !!python/object/apply:os.system ['rm -rf /']
```

If you load this with `yaml.load(f)`, it **executes** `os.system('rm -rf /')` during
parsing. This is not theoretical -- it is a known CVE (CVE-2017-18342) and has been
exploited in real systems.

`yaml.safe_load` restricts parsing to basic Python types only: strings, numbers,
lists, dicts, booleans, and `None`. No arbitrary object construction. Always use
`yaml.safe_load` unless you have a very specific reason and fully control the input.

**`return dict(yaml.safe_load(f))`** -- `yaml.safe_load` returns a Python object
matching the YAML structure. For our `params.yaml`, this is already a dict. Wrapping
it in `dict()` is a defensive measure -- if the YAML file were empty or contained
only a scalar, `yaml.safe_load` would return `None` or a string. `dict(None)` would
raise a `TypeError`, which is a clearer error than getting `None` back and having a
`NoneType has no attribute` error somewhere downstream.

#### get_project_root() -- Simple Accessor

```python
def get_project_root() -> Path:
    return PROJECT_ROOT
```

This is a simple accessor. Why not just import `PROJECT_ROOT` directly? Encapsulation.
If you later need to add logic (e.g., checking an environment variable override), you
change this one function instead of every import site. It also makes mocking easier in
tests -- you can patch `get_project_root` to return a temporary directory.

#### get_aws_config() -- Environment Variable Precedence

```python
def get_aws_config(params: dict | None = None) -> dict:
    if params is None:
        params = load_params()
    return {
        "region": os.getenv("AWS_REGION", params["aws"]["region"]),
        "s3_bucket": os.getenv("S3_BUCKET", params["aws"]["s3_bucket"]),
        "ecr_repository": params["aws"]["ecr_repository"],
        "lambda_function": params["aws"]["lambda_function"],
    }
```

**`os.getenv("AWS_REGION", params["aws"]["region"])`** -- This is the **environment
variable with fallback** pattern. `os.getenv(key, default)` returns:
1. The value of the environment variable `AWS_REGION` if it is set
2. The value from `params.yaml` (`params["aws"]["region"]` = `"us-east-1"`) if the
   environment variable is not set

**Why this precedence?** This is a standard 12-Factor App pattern. The priority is:
1. **Environment variables** (highest) -- set by CI/CD, Kubernetes, Docker, etc.
2. **Config file** (fallback) -- checked into version control as defaults

This lets you:
- Run locally with defaults from `params.yaml` (no env vars needed)
- Override in CI with `AWS_REGION=us-west-2 python -m src.data.ingest`
- Override in Kubernetes via ConfigMaps or Secrets
- Override in Docker via `-e AWS_REGION=eu-west-1`

**Real-world example: different configs for dev/staging/prod:**

```
# Developer laptop -- uses defaults from params.yaml
python -m src.data.ingest
# --> region = us-east-1 (from params.yaml)

# CI/CD staging pipeline -- env vars set in GitHub Actions
AWS_REGION=us-west-2 S3_BUCKET=staging-fraud-bucket python -m src.data.ingest
# --> region = us-west-2 (from env var, overrides params.yaml)

# Production Kubernetes pod -- env vars from ConfigMap
# The pod spec has: AWS_REGION=eu-west-1, S3_BUCKET=prod-fraud-bucket
# --> region = eu-west-1 (from env var)
```

Notice that `ecr_repository` and `lambda_function` do NOT have env var overrides.
This is intentional -- these rarely change between environments, so the complexity
of env var overrides is not worth it.

### The params.yaml File

```yaml
data:
  raw_path: data/raw/creditcard.csv
  processed_path: data/processed
  test_size: 0.2
  val_size: 0.1
  random_state: 42

features:
  drop_columns:
    - Time
  target_column: Class
  scaling_method: standard

model:
  type: xgboost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    ...

aws:
  region: us-east-1
  s3_bucket: mlops-fraud-detection-011015903780
  ecr_repository: mlops-fraud-detection
  lambda_function: mlops-fraud-prediction
```

This file is the single source of truth for ALL pipeline parameters. DVC uses it to
detect when parameters change and re-run affected stages. MLflow logs it for
experiment tracking. The structure is hierarchical -- `data`, `features`, `model`,
`aws` -- making it easy to find and modify settings. The fact that this is YAML (not
Python constants) means non-engineers (data scientists, product managers) can read
and even edit it.

---

## 2. Logging (logger.py)

**File:** `src/utils/logger.py`

### Full Source Code

```python
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

### Why Structured Logging Matters in Production ML

In production, you do not have the luxury of adding `print()` statements and
re-running. When your model starts returning bad predictions at 3 AM, logs are
your only window into what happened. Good logging answers:
- What data was the model trained on? (shape, fraud ratio, file path)
- Did validation pass? Were there warnings?
- What were the training metrics at each stage?
- When did the drift detection trigger?

Without structured, consistent logging, debugging a production ML failure is like
performing surgery in the dark.

### Line-by-Line Breakdown

```python
import logging
import sys
```

**`import logging`** -- Python's built-in logging module. It is part of the standard
library, so no pip install needed. The `logging` module implements a sophisticated
system with loggers, handlers, formatters, and filters organized in a hierarchy.

**`import sys`** -- The `sys` module provides access to system-specific parameters.
Here we use `sys.stdout` to direct log output to standard output.

#### The get_logger Function

```python
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
```

**`name: str`** -- The logger name. By convention, this is `__name__`, which Python
sets to the module's fully qualified name. For example:
- In `src/data/ingest.py`, `__name__` is `"src.data.ingest"`
- In `src/data/validate.py`, `__name__` is `"src.data.validate"`

**Why use `__name__`?** Because Python loggers form a **hierarchy** based on dot-
separated names. A logger named `"src.data.ingest"` is a child of `"src.data"`,
which is a child of `"src"`, which is a child of the **root** logger. This means:
- You can set the level for ALL data loggers with `logging.getLogger("src.data").setLevel(DEBUG)`
- You can set the level for the entire project with `logging.getLogger("src").setLevel(WARNING)`
- Each module's logs are tagged with its name, so you can grep for `[src.data.ingest]`

**`level: int = logging.INFO`** -- The default logging level. The hierarchy from
least to most severe is: `DEBUG` (10) < `INFO` (20) < `WARNING` (30) < `ERROR` (40)
< `CRITICAL` (50). At `INFO` level, you see everything except `DEBUG` messages.

#### logging.getLogger(name) and the Logger Registry

```python
    logger = logging.getLogger(name)
```

**This is NOT creating a new logger each time.** `logging.getLogger` maintains a
global registry (a dictionary) of loggers. If you call `getLogger("src.data.ingest")`
twice, you get the **exact same logger object** both times. This is the **Singleton
pattern** applied per name.

Why does this matter? Because multiple modules might import and call `get_logger`
for the same name. Without the registry, you would get duplicate loggers with
duplicate handlers, producing duplicate log lines.

#### The Handler Guard

```python
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
```

**`if not logger.handlers:`** -- This guard prevents the **duplicate handler
problem**. Here is what would happen without it:

1. `ingest.py` imports and calls `get_logger("src.data.ingest")` -- adds 1 handler
2. `validate.py` imports `ingest.py` (indirectly) and calls `get_logger("src.data.ingest")` again
3. Now the logger has 2 handlers, so every `logger.info()` call prints the message **twice**
4. In a big project with many imports, you might see the same log line 5+ times

The guard checks if the logger already has handlers attached. If it does, it skips
adding another one. Simple but essential.

**`logging.StreamHandler(sys.stdout)`** -- A **StreamHandler** sends log records to
a stream (file-like object). Here we use `sys.stdout` (standard output).

Why `sys.stdout` instead of the default `sys.stderr`? Because many log aggregation
systems (CloudWatch, Datadog, Kubernetes `kubectl logs`) capture stdout by default.
Also, in a DVC pipeline, stdout is captured and displayed, making it easier to see
pipeline progress.

**Other handler types you will encounter in production:**
- **`FileHandler`** -- writes to a file: `logging.FileHandler("pipeline.log")`
- **`RotatingFileHandler`** -- writes to a file, rotates when it gets too big (prevents disk full)
- **`CloudWatch handler`** (via `watchtower` library) -- sends logs directly to AWS CloudWatch
- **`SysLogHandler`** -- sends to the system's syslog daemon

**`logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")`** -- This
defines the format of each log line. The format codes are:
- **`%(asctime)s`** -- Timestamp like `2026-05-16 14:30:22,105`
- **`%(name)s`** -- Logger name like `src.data.ingest`
- **`%(levelname)s`** -- Level like `INFO`, `WARNING`, `ERROR`
- **`%(message)s`** -- The actual message you pass to `logger.info()`

A real log line looks like:
```
2026-05-16 14:30:22,105 [src.data.ingest] INFO: Data saved to data/raw/creditcard.csv -- shape: (284807, 31)
```

**`logger.addHandler(handler)`** -- Attaches the handler to the logger. A logger can
have multiple handlers (e.g., one for stdout, one for a file, one for CloudWatch).

```python
    logger.setLevel(level)
    return logger
```

**`logger.setLevel(level)`** -- Sets the minimum severity level. Messages below this
level are discarded. Note this is set **outside** the `if not logger.handlers` block,
so the level can be changed even if the handler already exists.

**Real-world debugging example:**

Imagine your fraud detection model starts flagging 40% of transactions as fraud
(normally it flags 0.2%). You SSH into the production server and check the logs:

```
2026-05-16 03:12:01 [src.data.validate] WARNING: Null values found: Amount    14892
2026-05-16 03:12:01 [src.data.validate] INFO: Fraud ratio: 0.0017
2026-05-16 03:12:03 [src.data.preprocess] INFO: Train fraud ratio: 0.0017
2026-05-16 03:15:44 [src.models.train] INFO: Training completed -- AUC: 0.97
```

Everything looks fine in training. But then you check the serving logs:

```
2026-05-16 06:00:01 [src.serving.app] ERROR: Scaler loaded from /old/path/scaler.pkl
2026-05-16 06:00:01 [src.serving.app] INFO: Scaler mean: [0.0], std: [1.0]
```

The scaler is loading from the wrong path and has identity transform values. The
model is receiving unscaled `Amount` values, which are orders of magnitude larger
than what it saw during training. The logger names tell you exactly which module
and which line to investigate.

---

## 3. Data Ingestion (ingest.py)

**File:** `src/data/ingest.py`

This module generates a synthetic credit card fraud dataset that mimics the famous
Kaggle dataset (284,807 transactions, ~0.17% fraud). In a real project, this would
download from Kaggle or an S3 bucket. The synthetic version ensures the pipeline
runs end-to-end even without credentials.

### Full Source Code

```python
"""Data ingestion: downloads the credit card fraud dataset.

Uses a synthetic generator as fallback when the Kaggle dataset is unavailable,
ensuring the pipeline always runs end-to-end.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_fraud_data(
    n_samples: int = 284807, fraud_ratio: float = 0.00173
) -> pd.DataFrame:
    """Generate synthetic data mimicking the Kaggle credit card fraud dataset structure."""
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit_features = rng.standard_normal((n_legit, 28))
    fraud_features = rng.standard_normal((n_fraud, 28)) * 1.5 + rng.uniform(-1, 1, (n_fraud, 28))

    features = np.vstack([legit_features, fraud_features])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    time_col = np.sort(rng.uniform(0, 172800, n_samples))
    amount_legit = rng.exponential(scale=88, size=n_legit)
    amount_fraud = rng.exponential(scale=122, size=n_fraud)
    amount = np.concatenate([amount_legit, amount_fraud])

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_col, features, amount, labels])

    shuffle_idx = rng.permutation(n_samples)
    data = data[shuffle_idx]

    return pd.DataFrame(data, columns=columns)


def ingest_data() -> Path:
    params = load_params()
    raw_path = Path(get_project_root() / params["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info(f"Raw data already exists at {raw_path}")
        return raw_path

    logger.info("Generating synthetic credit card fraud dataset")
    df = generate_synthetic_fraud_data()

    df.to_csv(raw_path, index=False)
    logger.info(f"Data saved to {raw_path} -- shape: {df.shape}")
    fraud_count = int(df["Class"].sum())
    logger.info(f"Fraud ratio: {df['Class'].mean():.4f} ({fraud_count} fraud / {len(df)} total)")

    return raw_path


if __name__ == "__main__":
    ingest_data()
```

### Line-by-Line Breakdown

#### Imports and Logger Setup

```python
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

**`logger = get_logger(__name__)`** -- This is called at module level (not inside a
function). `__name__` evaluates to `"src.data.ingest"`, so every log message from
this file is tagged with that name. Because it is at module level, the logger is
created once when the module is first imported and reused for all subsequent calls.

#### The Synthetic Data Generator

```python
def generate_synthetic_fraud_data(
    n_samples: int = 284807, fraud_ratio: float = 0.00173
) -> pd.DataFrame:
```

**`n_samples: int = 284807`** -- The exact number of rows in the real Kaggle credit
card fraud dataset. Using the same count means our pipeline's memory usage, timing,
and split sizes match what we would see with real data.

**`fraud_ratio: float = 0.00173`** -- 0.173% of transactions are fraudulent. This is
the real ratio from the Kaggle dataset. In the real world, credit card fraud rates
range from 0.1% to 0.3%. This extreme imbalance is what makes fraud detection
challenging -- a model that always predicts "not fraud" achieves 99.83% accuracy.

#### The Random Number Generator

```python
    rng = np.random.default_rng(42)
```

**`np.random.default_rng(42)`** creates a **Generator** object with seed 42.

**Generator vs legacy RandomState:**

Before NumPy 1.17, you would use `np.random.seed(42)` followed by
`np.random.normal(...)`. This used the **global RandomState** with the Mersenne
Twister algorithm. Problems with the old approach:
1. **Global state** -- `np.random.seed(42)` sets a global seed, so any library that
   calls `np.random` functions can interfere with your sequence.
2. **Not reproducible across threads** -- if two threads call `np.random.normal()`,
   the order is non-deterministic.
3. **Slower algorithm** -- Mersenne Twister has known statistical weaknesses.

**`default_rng(42)`** uses the new **Generator** API with the PCG64 algorithm:
1. **Local state** -- `rng` is a local variable, so nothing else can interfere.
2. **Better statistics** -- PCG64 has better statistical properties than Mersenne Twister.
3. **Faster** -- PCG64 is about 2x faster for generating random numbers.
4. **Reproducible** -- Same seed always produces the same sequence.

**Why seed 42?** It is an arbitrary convention (a reference to "The Hitchhiker's Guide
to the Galaxy" where 42 is the answer to life, the universe, and everything). Any
fixed seed works. The point is **reproducibility** -- everyone who runs this code gets
the exact same dataset, which means the exact same model metrics.

#### Calculating Split Sizes

```python
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
```

**`int(284807 * 0.00173)`** = `int(492.71611)` = `492`. The `int()` function
truncates (floors) the float. We get 492 fraud transactions and 284,315 legitimate
ones. The subtraction `n_samples - n_fraud` ensures the total is exactly `n_samples`.

#### Generating Features

```python
    legit_features = rng.standard_normal((n_legit, 28))
```

**`rng.standard_normal((n_legit, 28))`** generates a matrix of shape
`(284315, 28)` filled with values drawn from the **standard normal distribution**
(mean=0, standard deviation=1). Each row is one transaction, each column is one
of the V1-V28 features.

Why 28 features? The real Kaggle dataset has features V1 through V28, which are the
result of a **PCA (Principal Component Analysis)** transformation applied to the
original features (which were confidential). PCA outputs are, by construction, zero-
centered and approximately unit-variance -- which is why `standard_normal` is a
reasonable simulation.

The shape tuple `(n_legit, 28)` tells NumPy to create a 2D array. `rng.standard_normal`
is the Generator equivalent of `np.random.randn()` but with better statistical
properties.

```python
    fraud_features = rng.standard_normal((n_fraud, 28)) * 1.5 + rng.uniform(-1, 1, (n_fraud, 28))
```

This is the key line that makes fraud transactions **different** from legitimate ones.
Let us break down the math:

1. **`rng.standard_normal((n_fraud, 28))`** -- Start with a (492, 28) matrix of
   standard normal values (mean=0, std=1), same as legitimate.

2. **`* 1.5`** -- Multiply every value by 1.5. This **increases the variance**. The
   standard deviation goes from 1.0 to 1.5. Fraudulent transactions now have more
   "spread" in their feature values -- they are more erratic.

3. **`+ rng.uniform(-1, 1, (n_fraud, 28))`** -- Add a random **shift** between -1
   and 1 to each value. `rng.uniform(low, high, size)` draws from a uniform
   distribution on [low, high). This shifts the mean away from 0 -- fraud
   transactions are now centered at a slightly different location in feature space.

**What does this simulate?** In real fraud data, fraudulent transactions have:
- **Higher variance** -- fraudsters try different strategies, creating more diverse patterns
- **Shifted means** -- fraud patterns occupy a different region of the feature space than legitimate transactions
- **Some overlap** -- not all fraud is obviously different (hence `+rng.uniform(-1,1)` rather than `+5`)

This is a simplified version of what you see in the real data, where a classifier can
learn to separate the two classes but cannot achieve perfect separation.

#### Combining Features and Labels

```python
    features = np.vstack([legit_features, fraud_features])
    labels = np.array([0] * n_legit + [1] * n_fraud)
```

**`np.vstack([legit_features, fraud_features])`** -- **Vertical stack**. Stacks
arrays on top of each other (row-wise).

```
legit_features:  shape (284315, 28)
fraud_features:  shape (492, 28)
                 ---- vstack ----
features:        shape (284807, 28)
```

The legitimate rows come first, then the fraud rows. (We will shuffle later.)

**`np.array([0] * n_legit + [1] * n_fraud)`** -- Creates the label array.
`[0] * 284315` produces a Python list of 284,315 zeros. `[1] * 492` produces 492
ones. The `+` concatenates the two lists. `np.array(...)` converts to a NumPy array.
The order matches `features` -- first all legitimate (0), then all fraud (1).

#### Time Column

```python
    time_col = np.sort(rng.uniform(0, 172800, n_samples))
```

**`rng.uniform(0, 172800, n_samples)`** -- Generates 284,807 random values uniformly
distributed between 0 and 172,800. The number 172,800 is **48 hours in seconds**
(48 * 60 * 60 = 172,800). In the real Kaggle dataset, the `Time` column represents
seconds elapsed since the first transaction, and the data covers about 2 days.

**`np.sort(...)`** -- Sorts in ascending order. Transactions should be in
chronological order (or at least the Time column should be non-decreasing). Without
sorting, the Time values would be scattered randomly, which is unrealistic.

#### Amount Column -- Why Exponential Distribution?

```python
    amount_legit = rng.exponential(scale=88, size=n_legit)
    amount_fraud = rng.exponential(scale=122, size=n_fraud)
    amount = np.concatenate([amount_legit, amount_fraud])
```

**`rng.exponential(scale=88, size=n_legit)`** -- Draws from an **exponential
distribution** with mean (scale) 88.

**Why exponential?** Real credit card transaction amounts follow a heavily
right-skewed distribution:
- Most transactions are small ($5 coffee, $12 lunch, $50 groceries)
- A few are medium ($200 electronics, $500 flights)
- Very few are large ($5,000 jewelry, $10,000 luxury items)

The exponential distribution captures this skewness. The probability density function
is f(x) = (1/scale) * exp(-x/scale), which starts high at x=0 and decays
exponentially. With scale=88, the median is about $61, and most values are under $200,
but a few exceed $1,000.

**Why different scales for fraud (122) vs legitimate (88)?** Fraudulent transactions
tend to be slightly larger on average -- fraudsters try to extract maximum value
before the card is blocked. The mean fraud amount ($122) is about 39% higher than
the mean legitimate amount ($88). This matches patterns observed in real fraud data.

**`np.concatenate([amount_legit, amount_fraud])`** -- Joins the two 1D arrays
end-to-end. Unlike `vstack` (which stacks vertically in 2D), `concatenate` works
on 1D arrays by default.

```
amount_legit: shape (284315,)
amount_fraud: shape (492,)
              -- concatenate --
amount:       shape (284807,)
```

#### Assembling the Full Dataset

```python
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
```

**`[f"V{i}" for i in range(1, 29)]`** -- A list comprehension that generates
`["V1", "V2", "V3", ..., "V28"]`. The `f"V{i}"` is an f-string (formatted string
literal). `range(1, 29)` goes from 1 to 28 inclusive.

The full column list is:
`["Time", "V1", "V2", ..., "V28", "Amount", "Class"]` -- 31 columns total.

```python
    data = np.column_stack([time_col, features, amount, labels])
```

**`np.column_stack`** -- Stacks 1D and 2D arrays as columns. This is more flexible
than `np.hstack` because it handles 1D arrays correctly (treats them as columns, not
rows).

```
time_col:  shape (284807,)    --> treated as (284807, 1)
features:  shape (284807, 28)
amount:    shape (284807,)    --> treated as (284807, 1)
labels:    shape (284807,)    --> treated as (284807, 1)
                               -- column_stack --
data:      shape (284807, 31)  = 1 + 28 + 1 + 1
```

#### Shuffling

```python
    shuffle_idx = rng.permutation(n_samples)
    data = data[shuffle_idx]
```

**`rng.permutation(n_samples)`** -- Generates a random permutation of integers
from 0 to n_samples-1. For example, `rng.permutation(5)` might return
`array([3, 0, 4, 1, 2])`.

**`data[shuffle_idx]`** -- This is NumPy **fancy indexing** (also called advanced
indexing). It reorders the rows of `data` according to `shuffle_idx`. Row 3 becomes
the first row, row 0 becomes the second, etc.

**Why shuffle?** Because we stacked all legitimate transactions first, then all fraud.
If we do not shuffle, the first ~284,000 rows are all legitimate and the last ~492
are all fraud. This would cause problems with time-based splitting, batch training
(early batches see no fraud), and any analysis that looks at data in order.

Note that the `Time` column is now **out of order** after shuffling. This is fine
because we drop the `Time` column during preprocessing (it is in `drop_columns` in
`params.yaml`).

#### Creating the DataFrame

```python
    return pd.DataFrame(data, columns=columns)
```

**`pd.DataFrame(data, columns=columns)`** -- The DataFrame constructor accepts a 2D
NumPy array and a list of column names.

- `data` is a `(284807, 31)` NumPy array of type `float64`
- `columns` is a list of 31 strings

The constructor matches columns left-to-right: column 0 of the array gets the name
`"Time"`, column 1 gets `"V1"`, and so on. The DataFrame now has labeled columns and
integer row indices (0, 1, 2, ..., 284806).

Important: all values are `float64` because NumPy arrays are homogeneous. Even
the `Class` column is `0.0` and `1.0` (floats, not ints). This matters in the
validation step.

#### The ingest_data() Function

```python
def ingest_data() -> Path:
    params = load_params()
    raw_path = Path(get_project_root() / params["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
```

**`params["data"]["raw_path"]`** -- From `params.yaml`, this is
`"data/raw/creditcard.csv"`. It is a relative path.

**`get_project_root() / params["data"]["raw_path"]`** -- Joins the project root with
the relative path, producing something like
`/Users/manishsingh/Desktop/miss/miss/mlops/data/raw/creditcard.csv`.

**`raw_path.parent`** -- The parent directory:
`/Users/manishsingh/Desktop/miss/miss/mlops/data/raw/`

**`.mkdir(parents=True, exist_ok=True)`** -- Creates the directory, with two critical
parameters:

- **`parents=True`** -- If `data/` does not exist, create it AND `data/raw/` inside
  it. Without this, you would get a `FileNotFoundError` if the parent does not exist.
  It is like `mkdir -p` in Unix.

- **`exist_ok=True`** -- If the directory already exists, do nothing. Without this,
  you would get a `FileExistsError` on the second run. This makes the function
  **idempotent** -- you can run it 10 times and get the same result.

#### Caching / Short-Circuit

```python
    if raw_path.exists():
        logger.info(f"Raw data already exists at {raw_path}")
        return raw_path
```

**Idempotency check.** If the data file already exists, skip regeneration and return
immediately. This means:
- Running the pipeline twice does not waste time regenerating data
- DVC can skip this stage if the output is already cached
- You do not accidentally overwrite data that might have been manually edited

#### Saving the Data

```python
    df.to_csv(raw_path, index=False)
```

**`df.to_csv(raw_path, index=False)`** -- Saves the DataFrame as a CSV file.

**`index=False`** -- This is critical. By default, pandas adds the DataFrame's row
index as the first column in the CSV:

```csv
,Time,V1,V2,...   <-- with index=True (default)
0,43532.0,-1.234,...
1,12001.5,0.567,...
```

With `index=False`:
```csv
Time,V1,V2,...    <-- no index column
43532.0,-1.234,...
12001.5,0.567,...
```

The index is just `0, 1, 2, ...` which carries no information. Including it would
add a useless column that confuses downstream processing (validate.py would need to
account for it, the schema would need to include it, etc.).

```python
    fraud_count = int(df["Class"].sum())
    logger.info(f"Fraud ratio: {df['Class'].mean():.4f} ({fraud_count} fraud / {len(df)} total)")
```

**`df["Class"].sum()`** -- Since Class is 0 or 1, the sum equals the count of fraud
cases. We wrap in `int()` because `.sum()` returns a float (remember, Class is
`float64`), and printing `492.0` looks odd.

**`df['Class'].mean()`** -- For a binary column (0s and 1s), the mean IS the ratio
of 1s. Mean = sum / count = 492 / 284807 = 0.001728. This is a very common trick.

**`:.4f`** -- Format to 4 decimal places: `0.0017`.

#### Real-World Example: Replacing Synthetic Data with Kaggle Download

In a production version, you would replace `generate_synthetic_fraud_data()` with:

```python
import kaggle  # pip install kaggle

def download_kaggle_data():
    kaggle.api.dataset_download_files(
        'mlg-ulb/creditcardfraud',
        path=str(raw_path.parent),
        unzip=True
    )
```

Or download from S3:

```python
import boto3

def download_from_s3():
    s3 = boto3.client('s3')
    s3.download_file(
        aws_config['s3_bucket'],
        'data/raw/creditcard.csv',
        str(raw_path)
    )
```

The synthetic generator is the fallback for when neither is available -- it ensures
the pipeline can run on any machine for development and testing.

---

## 4. Data Validation (validate.py)

**File:** `src/data/validate.py`

Data validation is the gate between "data that arrived" and "data we trust to train
on." Without it, garbage data silently produces a garbage model, and you only
discover the problem when customers complain.

### Full Source Code

```python
"""Data validation: schema checks and statistical tests on ingested data."""

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from src.utils.config import get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_schema() -> DataFrameSchema:
    v_columns = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}
    return DataFrameSchema(
        columns={
            "Time": Column(float, Check.ge(0), nullable=False),
            **v_columns,
            "Amount": Column(float, Check.ge(0), nullable=False),
            "Class": Column(float, Check.isin([0.0, 1.0]), nullable=False),
        },
        coerce=True,
    )


def validate_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    params = load_params()
    if df is None:
        raw_path = get_project_root() / params["data"]["raw_path"]
        df = pd.read_csv(raw_path)

    schema = get_schema()
    validated_df = schema.validate(df, lazy=True)
    logger.info(f"Schema validation passed -- {len(validated_df)} rows")

    null_counts = validated_df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found")

    dup_count = validated_df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    fraud_ratio = validated_df["Class"].mean()
    logger.info(f"Fraud ratio: {fraud_ratio:.4f}")
    if fraud_ratio > 0.5:
        raise ValueError(f"Unexpected fraud ratio {fraud_ratio:.4f} -- data may be corrupted")

    return validated_df


if __name__ == "__main__":
    validate_data()
```

### Line-by-Line Breakdown

#### What is Pandera and Why Use It?

```python
from pandera import Check, Column, DataFrameSchema
```

**Pandera** is a data validation library for pandas DataFrames. It lets you define a
**schema** -- a formal specification of what your data should look like -- and then
validate any DataFrame against that schema.

**Pandera vs Great Expectations:**

| Feature           | Pandera                        | Great Expectations               |
|-------------------|--------------------------------|----------------------------------|
| Complexity        | Lightweight, code-first        | Heavy, JSON/YAML config-heavy    |
| Integration       | Works inline in Python code    | Separate validation layer        |
| Learning curve    | Minutes                        | Days                             |
| Best for          | ML pipelines, data science     | Data engineering, large teams    |
| Schema as code    | Yes, Pythonic                  | Yes, but verbose                 |
| Type checking     | Pandas dtypes + custom checks  | Extensive built-in expectations  |

For an ML pipeline where you want quick, inline validation, Pandera is the right
choice. Great Expectations is more suited for data platform teams managing thousands
of tables.

#### Defining the Schema

```python
def get_schema() -> DataFrameSchema:
    v_columns = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}
```

**`{f"V{i}": Column(float, nullable=False) for i in range(1, 29)}`** -- A dict
comprehension that creates 28 column definitions. It produces:

```python
{
    "V1": Column(float, nullable=False),
    "V2": Column(float, nullable=False),
    ...
    "V28": Column(float, nullable=False),
}
```

**`Column(float, nullable=False)`** -- Each V column must:
- Be of type `float` (or coercible to float)
- Not contain any null/NaN values (`nullable=False`)

No range checks are applied to V columns because PCA outputs can be any real number
(positive or negative, large or small). Adding arbitrary bounds would cause false
validation failures.

```python
    return DataFrameSchema(
        columns={
            "Time": Column(float, Check.ge(0), nullable=False),
            **v_columns,
            "Amount": Column(float, Check.ge(0), nullable=False),
            "Class": Column(float, Check.isin([0.0, 1.0]), nullable=False),
        },
        coerce=True,
    )
```

**`"Time": Column(float, Check.ge(0), nullable=False)`** -- The Time column must be:
- `float` type
- Greater than or equal to 0 (`Check.ge(0)` -- "ge" = "greater or equal")
- Not nullable

Time represents seconds since the first transaction, so negative values are
impossible. This catches data corruption like negative timestamps.

**`**v_columns`** -- The `**` unpacks the dictionary into the parent dictionary. This
is the dict spread operator. The result is that all 28 V columns are inserted
alongside Time, Amount, and Class.

**`"Amount": Column(float, Check.ge(0), nullable=False)`** -- Transaction amounts
must be non-negative. A negative transaction amount would indicate a data error
(refunds might exist in real data, but our synthetic data does not include them).

**`"Class": Column(float, Check.isin([0.0, 1.0]), nullable=False)`**

**`Check.isin([0.0, 1.0])`** -- The Class column can only contain 0.0 or 1.0.

**Why `0.0` and `1.0` (floats) instead of `0` and `1` (ints)?** Because the data
comes from a CSV file read by `pd.read_csv`. In the CSV, the Class column contains
values like `0.0` and `1.0` (they went through NumPy float64 during generation).
When pandas reads them, it infers the column type as `float64`. If we checked for
integer `0` and `1`, the validation would fail because `0.0 != 0` in a strict type
check. The `coerce=True` flag (explained below) mitigates this somewhat, but using
the actual values makes the schema more explicit.

**`coerce=True`** -- Tells Pandera to attempt type coercion before validation. If a
column contains string `"1.0"`, Pandera will try to convert it to `float(1.0)` before
checking. Without coercion, type mismatches would cause validation failures even when
the data is logically correct. This is defensive programming -- CSV files lose type
information, so columns that were `int` when saved might be `object` when loaded.

#### The validate_data Function

```python
def validate_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    params = load_params()
    if df is None:
        raw_path = get_project_root() / params["data"]["raw_path"]
        df = pd.read_csv(raw_path)
```

**Dual-mode function:** It can either:
1. Accept a DataFrame directly (when called from `preprocess.py` which already has
   the data in memory)
2. Load from disk (when called standalone via `__main__`)

This avoids redundant disk reads when used in a pipeline but still works as a
standalone script.

#### Schema Validation with Lazy Mode

```python
    schema = get_schema()
    validated_df = schema.validate(df, lazy=True)
```

**`schema.validate(df, lazy=True)`** -- This is the core validation call.

**`lazy=True`** is the key parameter. It controls error reporting behavior:

**Without `lazy=True` (fail-fast mode):**
- Validation stops at the FIRST error
- Raises `SchemaError` with one error message
- You fix that error, re-run, find the next error, fix it, re-run...
- Frustrating when there are 10 errors

**With `lazy=True` (collect-all mode):**
- Validation checks EVERYTHING, collecting all errors
- Raises `SchemaErrors` (plural!) with a summary of ALL failures
- You see all 10 errors at once and fix them all in one pass

Example output with `lazy=True`:

```
pandera.errors.SchemaErrors:
Schema errors found:
  Column    Check          Index
  Amount    greater_or_equal_to(0)   [142, 8903, 50021]
  Class     isin([0.0, 1.0])        [7744]
  V15       not_nullable             [200103, 200104]
```

In a production pipeline, `lazy=True` is essential because:
1. You want to see ALL data quality issues, not just the first one
2. You can generate a comprehensive data quality report
3. You can decide which issues are critical vs which are warnings

#### Null Value Check

```python
    null_counts = validated_df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found")
```

**`validated_df.isnull()`** -- Returns a DataFrame of the same shape, with `True`
where the value is `NaN`/`None` and `False` otherwise. For a DataFrame with 284,807
rows and 31 columns, this produces a 284,807 x 31 boolean DataFrame.

**`.sum()`** -- Sums each column. Since `True=1` and `False=0`, this gives the count
of nulls per column. Result is a Series like:

```
Time      0
V1        0
V2        0
...
Amount    0
Class     0
dtype: int64
```

**`.any()`** -- Returns `True` if ANY value in the Series is truthy (non-zero). This
is the efficient way to check "are there any nulls anywhere?"

**`null_counts[null_counts > 0]`** -- Boolean indexing. Only shows columns that have
nulls, filtering out the zeros. This is much more useful than printing all 31 columns
when only 2 have nulls.

**Why check for nulls separately from the schema?** The schema's `nullable=False`
would catch nulls too, but this separate check provides a WARNING (does not stop the
pipeline) with detailed counts. The schema check is pass/fail; this check is
informational.

#### Duplicate Row Check

```python
    dup_count = validated_df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows")
    else:
        logger.info("No duplicate rows found")
```

**`validated_df.duplicated()`** -- Returns a boolean Series. For each row, it checks
if the EXACT same combination of all column values has appeared in a previous row.
The first occurrence is `False`; subsequent duplicates are `True`.

**`.sum()`** -- Counts the number of `True` values (duplicate rows).

**Why check for duplicates?** Duplicate rows in training data are a subtle problem:
- They give extra weight to those data points during training
- If the same row appears in both train and test sets after splitting, you get
  **data leakage** (the model has "seen" test data during training)
- Duplicates often indicate a bug in the data pipeline (e.g., a table was joined
  incorrectly, or data was ingested twice)

#### Fraud Ratio Sanity Check

```python
    fraud_ratio = validated_df["Class"].mean()
    logger.info(f"Fraud ratio: {fraud_ratio:.4f}")
    if fraud_ratio > 0.5:
        raise ValueError(f"Unexpected fraud ratio {fraud_ratio:.4f} -- data may be corrupted")
```

**`validated_df["Class"].mean()`** -- For a column of 0s and 1s, the mean equals the
proportion of 1s. This is basic math: `mean = sum / count = (number of 1s) / (total rows)`.

For our data: `492 / 284807 = 0.001728`

**`if fraud_ratio > 0.5`** -- A hard sanity check. If more than 50% of transactions
are fraud, something is catastrophically wrong. Real fraud rates are 0.1-0.3%. Even
a 5% fraud rate would be alarming. The 0.5 threshold is deliberately generous -- it
catches only the most extreme corruption (like labels being inverted, or the dataset
being replaced with a test fixture).

**Why `raise ValueError` instead of just warning?** Because training a model on
corrupted data is worse than not training at all. A corrupt model in production can
approve fraudulent transactions or block legitimate ones. This check is a
**circuit breaker** -- it stops the pipeline before damage is done.

#### Real-World Example: Data Validation Catching a Production Bug

A real scenario from a payments company:

The data pipeline ran nightly, pulling transactions from a database. One night, a
database migration changed the `amount` column from cents (integer) to dollars
(decimal). The amounts dropped by 100x:

```
Before migration: amount = 8850   (meaning $88.50)
After migration:  amount = 88.50  (already in dollars)
```

Without validation, the model would have trained on data where "large" transactions
were 100x smaller. The model would learn that $88 is a huge transaction and flag
normal $200 purchases as fraud.

With Pandera schema validation, the team had a check like:
```python
"Amount": Column(float, Check.in_range(0, 100000), nullable=False)
```

The validation passed (amounts were still in range), but they also had a **statistical
check** on the mean amount. The mean dropped from ~$88 to ~$0.88, which triggered
an alert. They caught the bug before the model retrained.

This is why validation includes both schema checks (type, range) AND statistical
checks (fraud ratio, null counts) -- schema alone is not enough.

---

## 5. Data Preprocessing (preprocess.py)

**File:** `src/data/preprocess.py`

Preprocessing transforms validated raw data into the format the model needs:
splitting into train/val/test, feature engineering, and scaling. Every decision
here directly impacts model performance and, critically, whether your evaluation
metrics are trustworthy.

### Full Source Code

```python
"""Feature engineering and train/val/test splitting."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.validate import validate_data
from src.utils.config import get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data() -> dict[str, Path]:
    params = load_params()
    root = get_project_root()

    raw_path = root / params["data"]["raw_path"]
    processed_dir = root / params["data"]["processed_path"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = validate_data(df)

    drop_cols = params["features"]["drop_columns"]
    target_col = params["features"]["target_column"]

    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    relative_val_size = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=params["data"]["random_state"],
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    scale_cols = ["Amount"]
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    output_files = {}
    for name, data in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]:
        path = processed_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        output_files[name] = path

    scaler_path = processed_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    output_files["scaler"] = scaler_path

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
    logger.info(f"Scaler saved to {scaler_path}")

    return output_files


if __name__ == "__main__":
    preprocess_data()
```

### Line-by-Line Breakdown

#### Imports

```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data.validate import validate_data
```

- **`joblib`** -- A library for serializing Python objects to disk. Optimized for
  large NumPy arrays (which sklearn objects contain internally). More on this below.
- **`train_test_split`** -- sklearn's utility for splitting data into random subsets.
- **`StandardScaler`** -- sklearn's z-score normalization transformer.
- **`validate_data`** -- Our own validation function. Note how preprocessing
  *depends on* validation -- the pipeline is: ingest -> validate -> preprocess.

#### Setup and Loading

```python
def preprocess_data() -> dict[str, Path]:
    params = load_params()
    root = get_project_root()

    raw_path = root / params["data"]["raw_path"]
    processed_dir = root / params["data"]["processed_path"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = validate_data(df)
```

**`-> dict[str, Path]`** -- Returns a dictionary mapping names (like `"X_train"`,
`"scaler"`) to file paths. This allows the caller to know where each output was saved.

**`df = validate_data(df)`** -- Passes the loaded DataFrame to validation. This is
the pipeline pattern: each stage validates its input before processing. If validation
fails, preprocessing stops immediately. The validated DataFrame is returned (same
object, but now we know it passed checks).

#### Feature/Target Separation

```python
    drop_cols = params["features"]["drop_columns"]   # ["Time"]
    target_col = params["features"]["target_column"]  # "Class"

    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col].astype(int)
```

**`drop_cols + [target_col]`** -- `["Time"] + ["Class"]` = `["Time", "Class"]`. These
columns are removed from the feature matrix `X`.

**Why drop Time?** The `Time` column represents seconds since the first transaction.
In the real Kaggle dataset, this is not a useful feature for fraud detection because:
1. Fraud can happen at any time
2. The relationship between time and fraud is better captured by derived features
   (hour of day, day of week) which would be feature engineering
3. Including raw time would cause the model to overfit to the specific time window
   in the training data

**`df.drop(columns=drop_cols + [target_col])`** -- The `columns` parameter (as
opposed to the older `axis=1` syntax) makes the intent clear. This returns a NEW
DataFrame with those columns removed; the original `df` is unchanged.

**`df[target_col].astype(int)`** -- Converts Class from `float64` (0.0, 1.0) to
`int64` (0, 1). This is important because:
1. Classification targets should be integers (sklearn expects this for stratify)
2. Metrics like `classification_report` display class labels as integers
3. It prevents subtle bugs where `1.0 != 1` in certain comparisons

`X` now has shape `(284807, 29)` -- V1 through V28 plus Amount.
`y` now has shape `(284807,)` -- integer 0s and 1s.

#### First Split: Train+Val vs Test

```python
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],   # 0.2
        random_state=params["data"]["random_state"],  # 42
        stratify=y,
    )
```

**`train_test_split(X, y, ...)`** -- Splits both `X` and `y` simultaneously, keeping
rows aligned. It shuffles the data randomly, then takes the last 20% as test and the
first 80% as train+val.

**`test_size=0.2`** -- 20% of the data goes to the test set. For 284,807 rows, that
is approximately 56,961 test rows and 227,846 train+val rows.

**`random_state=42`** -- Seeds the random number generator used for shuffling. This
ensures the same split every time. Without it, each run would produce a different
split, making experiments non-reproducible. Two scientists comparing results need the
same split to have a fair comparison.

**`stratify=y`** -- This is CRITICAL for imbalanced data. Here is what it does:

**Without stratify (simple random split):**
The 492 fraud cases are randomly distributed. By chance, the test set might get 80
fraud cases (0.14%) while training gets 412 (0.18%). Or worse, test gets 120 (0.21%)
and training gets 372 (0.16%). The fraud ratios differ between splits, which means
your model's performance on the test set is not representative of its performance on
training-like data.

**With `stratify=y`:**
The split preserves the exact class ratio in each subset. If the original data has
0.173% fraud, then:
- Train+val: ~0.173% fraud (approximately 394 fraud out of 227,846)
- Test: ~0.173% fraud (approximately 98 fraud out of 56,961)

The algorithm groups rows by class, then samples proportionally from each group. This
is called **stratified sampling**.

**Why is this critical?** With only 492 fraud cases total, random splitting could
easily leave very few fraud cases in the test set. If test has only 50 fraud cases
instead of 98, your precision/recall estimates have high variance and are unreliable.
Stratification guarantees representative subsets.

#### Second Split: Train vs Validation -- The Math

```python
    relative_val_size = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=params["data"]["random_state"],
        stratify=y_train_val,
    )
```

This is the trickiest math in the file. Let us work through it step by step.

**Goal:** We want three splits:
- Train: 70% of original data
- Validation: 10% of original data
- Test: 20% of original data

**Problem:** `train_test_split` only splits into TWO sets. So we split twice:
1. First split: 80% train+val, 20% test (done above)
2. Second split: Split the 80% into train and val

**The formula:**

```python
relative_val_size = params["data"]["val_size"] / (1 - params["data"]["test_size"])
                  = 0.1 / (1 - 0.2)
                  = 0.1 / 0.8
                  = 0.125
```

**Why 0.125?** Because the validation set needs to be 10% of the ORIGINAL data, but
we are splitting the REMAINING 80%. So the validation set needs to be:

```
10% of original = what fraction of the remaining 80%?
0.10 / 0.80 = 0.125 = 12.5% of the remaining data
```

Verification:
- Original: 284,807 rows
- After first split: 227,846 train+val, 56,961 test
- After second split: 227,846 * 0.875 = 199,365 train, 227,846 * 0.125 = 28,481 val
- Check: 199,365 / 284,807 = 70.0% (train)
- Check: 28,481 / 284,807 = 10.0% (val)
- Check: 56,961 / 284,807 = 20.0% (test)

The math works out to the exact 70/10/20 split we wanted.

**Why not split into three sets at once?** Sklearn's `train_test_split` does not
natively support three-way splits. You could use `np.split` or write custom logic,
but the two-step approach is the standard pattern and preserves stratification at
each step.

#### StandardScaler -- The Core of Feature Scaling

```python
    scaler = StandardScaler()
    scale_cols = ["Amount"]
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
```

**What is StandardScaler?**

StandardScaler applies **z-score normalization** (also called standardization):

```
z = (x - mean) / std
```

For each value, it subtracts the mean and divides by the standard deviation. After
transformation:
- The column has mean approximately 0
- The column has standard deviation approximately 1

**Example with Amount column:**
- Raw Amount values: [88.50, 12.30, 2450.00, 5.60, 150.00, ...]
- Mean = ~88, Std = ~250
- After scaling: [0.002, -0.303, 9.448, -0.330, 0.248, ...]

**Why scale?** Many ML algorithms (logistic regression, SVM, neural networks, KNN)
are sensitive to feature scale. If Amount ranges from 0 to 25,000 while V1 ranges
from -5 to 5, the algorithm will disproportionately weight Amount. Scaling puts all
features on the same "playing field."

**XGBoost (which this project uses) is actually NOT sensitive to feature scale**
because it uses decision trees, which split on thresholds regardless of scale.
However, scaling Amount is still good practice because:
1. It makes the data consistent if you switch models later
2. It helps with visualization and interpretation
3. Some downstream operations (like computing distances for drift detection) benefit

**Why only scale Amount and not V1-V28?**

The V1-V28 features came from PCA (Principal Component Analysis). PCA output is
ALREADY standardized -- by construction, PCA components have mean 0 and are
orthogonal. The real Kaggle dataset explicitly states that V1-V28 are PCA-
transformed. Scaling them again would not hurt (StandardScaler on already-standard
data is approximately an identity operation), but it would be unnecessary computation.

Amount, however, is the raw transaction amount in dollars and has a heavy-tailed
distribution (mean ~88, std ~250, range 0 to 25,691). It NEEDS scaling.

**StandardScaler vs MinMaxScaler vs RobustScaler:**

| Scaler         | Formula                      | When to use                          |
|----------------|------------------------------|--------------------------------------|
| StandardScaler | (x - mean) / std             | Normally distributed features        |
| MinMaxScaler   | (x - min) / (max - min)      | Bounded features, neural networks    |
| RobustScaler   | (x - median) / IQR           | Features with many outliers          |

For Amount, `RobustScaler` might actually be better because the exponential
distribution has outliers. But `StandardScaler` is the standard choice and works
well enough here.

#### The Critical Concept: fit_transform vs transform

```python
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
```

This is one of the most important concepts in all of ML engineering. Let us break
down exactly what happens:

**`scaler.fit_transform(X_train[scale_cols])`** does TWO things:
1. **fit**: Computes the mean and standard deviation FROM the training data.
   `scaler.mean_` = mean of Amount in X_train, `scaler.scale_` = std of Amount in
   X_train. These are stored as attributes of the scaler object.
2. **transform**: Applies the formula `(x - mean) / std` to the training data using
   the statistics just computed.

**`scaler.transform(X_val[scale_cols])`** does ONLY ONE thing:
- **transform**: Applies `(x - mean) / std` using the mean and std ALREADY stored
  from the fit step (i.e., the TRAINING data's statistics).

**Why NOT `scaler.fit_transform(X_val[...])`?**

This is the **data leakage** problem. If you fit the scaler on the validation or test
data, you are using information from those sets to transform them. This means:
1. The val/test distributions are centered at their own mean (not the training mean)
2. Your val/test metrics are biased -- they look better than they would in production
3. In production, you only have training statistics available

**Think of it this way:** In production, when a new transaction comes in, you do not
know the mean of all future transactions. You only know the mean from the data you
trained on. So you must use `scaler.transform()` with training statistics.

**The train/test leakage disaster -- a real-world example:**

A team at a fintech company built a fraud detection model. Their preprocessing was:

```python
# WRONG -- DO NOT DO THIS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data including test
X_train, X_test = train_test_split(X_scaled, ...)
```

What went wrong:
1. The scaler learned the mean and std from ALL data (train + test)
2. The test data's mean was "baked into" the transformation
3. Model achieved 0.99 AUC on test -- looked amazing
4. Deployed to production -- AUC dropped to 0.85
5. They lost 3 months of work and $500K in false approvals

The correct order is ALWAYS:
1. Split first
2. Fit on training only
3. Transform everything using training statistics

This code does it correctly: split happens on lines 34-49, then fit_transform on
line 53 (train only), then transform on lines 54-55 (val and test).

#### Saving the Processed Data

```python
    output_files = {}
    for name, data in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]:
        path = processed_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        output_files[name] = path
```

This loop saves six CSV files to `data/processed/`:
- `X_train.csv`, `X_val.csv`, `X_test.csv` -- feature matrices
- `y_train.csv`, `y_val.csv`, `y_test.csv` -- label vectors

Each is saved with `index=False` (no row numbers). The `output_files` dictionary
maps names to paths for the caller.

**Why separate X and y files?** This is a common pattern because:
1. The training step needs X_train and y_train together
2. The evaluation step needs X_test and y_test together
3. DVC can track each file independently and cache them separately
4. It is easier to inspect (you can open X_train.csv in Excel to check features)

#### Serializing the Scaler with joblib

```python
    scaler_path = processed_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    output_files["scaler"] = scaler_path
```

**`joblib.dump(scaler, scaler_path)`** -- Serializes the fitted `StandardScaler`
object to a `.pkl` (pickle) file.

**What is serialization?** It is the process of converting a Python object in memory
into a byte stream that can be written to disk. Later, you can **deserialize**
(load) it back into memory with `joblib.load(scaler_path)`.

**What is stored inside the scaler?** After `fit_transform`, the scaler contains:
- `scaler.mean_` -- the mean of the Amount column from training data
- `scaler.scale_` -- the standard deviation of the Amount column from training data
- `scaler.var_` -- the variance
- `scaler.n_samples_seen_` -- number of samples used to compute statistics
- `scaler.n_features_in_` -- number of features (1 in this case)
- `scaler.feature_names_in_` -- column names (["Amount"])

All of this is saved to disk so that during prediction (in production), we can
load the exact same scaler and apply the exact same transformation.

**Why joblib instead of pickle?**

| Feature        | joblib                             | pickle (stdlib)                  |
|----------------|------------------------------------|----------------------------------|
| NumPy arrays   | Optimized (memory-mapped, fast)    | Generic (slow for large arrays)  |
| File size      | Compressed (smaller files)         | Uncompressed                     |
| Speed          | Faster for sklearn objects         | Slower                           |
| Compatibility  | sklearn recommended                | General purpose                  |

The sklearn documentation explicitly recommends joblib over pickle for sklearn
objects because sklearn estimators contain NumPy arrays internally, and joblib
handles those much more efficiently.

**Security note:** Loading a pickle/joblib file executes arbitrary code. Never load
a `.pkl` file from an untrusted source. In production, the scaler file should be
stored in a controlled location (S3 with access controls, not a public URL).

#### Logging the Results

```python
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
    logger.info(f"Scaler saved to {scaler_path}")
```

These log lines produce output like:
```
2026-05-16 14:35:01 [src.data.preprocess] INFO: Train: (199365, 29), Val: (28481, 29), Test: (56961, 29)
2026-05-16 14:35:01 [src.data.preprocess] INFO: Train fraud ratio: 0.0017
2026-05-16 14:35:01 [src.data.preprocess] INFO: Scaler saved to /Users/.../data/processed/scaler.pkl
```

**`y_train.mean():.4f`** -- Confirms the fraud ratio is preserved after splitting
(stratification worked). If this showed 0.0000 or 0.5000, something went wrong.

**`X_train.shape`** -- Prints `(rows, columns)`. This is a quick sanity check that
the split sizes are reasonable.

---

## Summary: The Complete Data Pipeline Flow

```
params.yaml                    (configuration)
    |
    v
ingest.py                      (generate/download raw data)
    |  produces: data/raw/creditcard.csv (284807 rows, 31 columns)
    v
validate.py                    (schema + statistical checks)
    |  confirms: types, ranges, no nulls, fraud ratio sane
    v
preprocess.py                  (split + scale)
    |  produces:
    |    data/processed/X_train.csv  (199365 x 29)
    |    data/processed/X_val.csv    (28481 x 29)
    |    data/processed/X_test.csv   (56961 x 29)
    |    data/processed/y_train.csv  (199365 x 1)
    |    data/processed/y_val.csv    (28481 x 1)
    |    data/processed/y_test.csv   (56961 x 1)
    |    data/processed/scaler.pkl   (fitted StandardScaler)
    v
[ready for model training]
```

Every step is:
- **Configurable** via `params.yaml`
- **Logged** with structured messages
- **Validated** before passing data downstream
- **Reproducible** via fixed random seeds
- **Idempotent** -- safe to run multiple times

These properties are what distinguish an MLOps pipeline from a Jupyter notebook.

---

## Key Interview Takeaways

1. **Configuration:** Use YAML with env var overrides. Never hardcode paths or hyperparameters. `yaml.safe_load` prevents code execution attacks.

2. **Logging:** Use `logging.getLogger(__name__)` with the handler guard. Module-level loggers with hierarchical names. Never use `print()` in production.

3. **Data Generation:** `np.random.default_rng` (Generator API) over `np.random.seed` (legacy). Seed everything for reproducibility.

4. **Validation:** Schema validation with Pandera (or Great Expectations). Lazy mode to collect all errors. Statistical sanity checks beyond schema.

5. **Preprocessing:** Split BEFORE scaling. `fit_transform` on train, `transform` on val/test. Serialize the scaler for production use.

6. **The cardinal sin:** Fitting ANYTHING on test data. Whether it is a scaler, an encoder, or imputer statistics -- always fit on training data only.
