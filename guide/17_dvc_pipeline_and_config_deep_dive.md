# Guide 17: DVC Pipeline and Configuration Deep Dive

## Every Configuration File, Every Field, Every Design Decision Explained

This guide tears apart every configuration file in the project -- params.yaml, dvc.yaml,
dvc.lock, .dvc/config, requirements.txt, pyproject.toml, and .gitignore -- line by line.
By the end, you will understand not just WHAT each line does, but WHY it exists and how
it connects to the rest of the MLOps system.

---

## 1. params.yaml -- The Central Configuration Hub

params.yaml is the single source of truth for every tunable knob in the pipeline. DVC reads
this file to decide when stages need rerunning, MLflow logs these values as experiment
parameters, and every script in `src/` loads its configuration from here.

```yaml
# FILE: params.yaml (complete)

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
    min_child_weight: 3
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 50
    eval_metric: aucpr
    random_state: 42
    n_jobs: -1

training:
  experiment_name: fraud-detection
  cv_folds: 5
  early_stopping_rounds: 20

thresholds:
  min_recall: 0.02
  min_precision: 0.50
  min_f1: 0.03
  min_auc_roc: 0.90

monitoring:
  drift_threshold: 0.05
  performance_window_days: 7
  alert_on_drift: true

aws:
  region: us-east-1
  s3_bucket: mlops-fraud-detection-011015903780
  ecr_repository: mlops-fraud-detection
  lambda_function: mlops-fraud-prediction
  api_name: fraud-detection-api
```

### 1.1 The `data` Section

```yaml
data:
  raw_path: data/raw/creditcard.csv       # Where raw data lives after ingestion
  processed_path: data/processed           # Directory for all processed outputs
  test_size: 0.2                           # 20% of data reserved for final testing
  val_size: 0.1                            # 10% of data reserved for validation
  random_state: 42                         # Fixed seed for reproducibility
```

**`raw_path: data/raw/creditcard.csv`**

This is a relative path from the project root. The ingest script downloads the credit card
fraud dataset from Kaggle and writes it to this exact location. Every downstream stage
references this path. If you changed this value, DVC would detect the parameter change and
rerun the ingest stage.

The convention `data/raw/` vs `data/processed/` is a standard ML project structure inspired
by the Cookiecutter Data Science template. Raw data is immutable -- you never modify it in
place. Processed data is derived and regenerable.

**`test_size: 0.2` -- The 80/20 Rule**

This means 20% of the total dataset is held out as the test set. The 80/20 split comes from
the Pareto principle and decades of empirical practice in statistics. Here is the reasoning:

- Too small a test set (e.g., 5%): high variance in evaluation metrics. Your precision might
  be 0.95 on one run and 0.82 on another just because of which samples landed in the test set.
- Too large a test set (e.g., 50%): starves the model of training data. The model underfits
  because it never sees enough examples.
- 20% is the sweet spot for medium-to-large datasets. With 284,807 transactions in the credit
  card dataset, 20% gives us approximately 56,961 test samples -- more than enough for
  statistically stable metrics.

Alternative splits used in practice:
- 70/15/15 (train/val/test) for medium datasets
- 90/5/5 for very large datasets (millions of rows)
- 60/20/20 when you want extra validation stability

In this project: 70% train, 10% validation, 20% test (because val_size is taken from the
remaining 80% after the test split).

**`val_size: 0.1` -- The Validation Set**

The validation set serves a different purpose than the test set:
- **Validation set**: used DURING training to tune hyperparameters and monitor for overfitting.
  Early stopping watches validation loss to decide when to stop training.
- **Test set**: used AFTER training is finalized. Touched exactly once to get final metrics.
  Never used to make modeling decisions.

With 0.1 of the remaining 80%, the actual validation set is about 10% of total data, roughly
28,480 samples. This feeds into XGBoost's early stopping mechanism via `eval_set`.

**`random_state: 42` -- Reproducibility in ML**

The number 42 is a convention (a nod to The Hitchhiker's Guide to the Galaxy). Any fixed
integer would work. What matters is that it is FIXED, not what value it is.

What happens under the hood:
```python
from sklearn.model_selection import train_test_split

# WITH fixed seed -- same split every time
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
# Run 1: rows [5, 12, 99, ...] in test
# Run 2: rows [5, 12, 99, ...] in test  <-- IDENTICAL

# WITHOUT fixed seed -- different split every time
X_train, X_test = train_test_split(X, test_size=0.2)
# Run 1: rows [5, 12, 99, ...] in test
# Run 2: rows [3, 47, 201, ...] in test  <-- DIFFERENT
```

Without a fixed seed:
- You cannot reproduce an experiment. Your colleague runs the pipeline and gets different
  metrics. Was it the code change or the random split?
- DVC cannot detect "nothing changed." Different random splits mean different data files,
  which means different hashes, which means DVC reruns everything unnecessarily.
- Debugging becomes a nightmare. A bug that appears in one run disappears in the next because
  the problematic data point moved between train and test.

### 1.2 The `features` Section

```yaml
features:
  drop_columns:
    - Time
  target_column: Class
  scaling_method: standard
```

**`drop_columns: [Time]` -- Why Drop the Time Column?**

The credit card fraud dataset has a `Time` column that represents the number of seconds
elapsed between each transaction and the first transaction in the dataset. Here is why we
drop it:

1. **It is not a meaningful feature.** The absolute elapsed time (e.g., 4000 seconds from
   the first transaction) does not carry fraud signal. Transaction 4000 seconds in is not
   inherently more or less fraudulent.
2. **It leaks temporal ordering.** If you train on early transactions and test on later ones,
   the model could learn "high Time values = test set" rather than learning fraud patterns.
3. **The V1-V28 features already capture temporal patterns.** These are PCA-transformed
   features from the original dataset. Any useful temporal information is already encoded.

If you wanted to USE time-based features, you would engineer them differently:
- Hour of day (cyclical encoding)
- Day of week
- Time since last transaction by the same cardholder
These are meaningful. Raw elapsed seconds is not.

**`target_column: Class`**

This tells the pipeline which column is the label. `Class` is binary:
- 0 = legitimate transaction (284,315 samples, 99.83%)
- 1 = fraudulent transaction (492 samples, 0.17%)

This extreme imbalance is why `scale_pos_weight: 50` exists in the model config (explained
below).

**`scaling_method: standard`**

StandardScaler transforms each feature to have mean=0 and standard deviation=1:
```
X_scaled = (X - mean) / std_deviation
```

Why `standard` and not other methods?

| Method | Formula | Best When |
|--------|---------|-----------|
| `standard` (StandardScaler) | (X - mean) / std | Features are roughly Gaussian. Most general. |
| `minmax` (MinMaxScaler) | (X - min) / (max - min) | You need values in [0, 1]. Neural networks. |
| `robust` (RobustScaler) | (X - median) / IQR | Data has many outliers. Uses median, not mean. |
| `none` | No scaling | Tree-based models (they split on thresholds, scale-invariant). |

Technically, XGBoost does NOT need feature scaling -- it is tree-based and makes split
decisions based on feature value thresholds, not distances. So why scale here? Two reasons:
1. The pipeline is designed to be model-agnostic. If you swap XGBoost for logistic regression
   or a neural network, scaling is already handled.
2. The V1-V28 features from PCA are already standardized, but `Amount` is not. Scaling
   ensures consistency across all features.

### 1.3 The `model` Section

```yaml
model:
  type: xgboost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    min_child_weight: 3
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 50
    eval_metric: aucpr
    random_state: 42
    n_jobs: -1
```

**How the code reads this section:**

```python
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

model_params = params["model"]["params"]
# model_params is now a Python dict:
# {
#     "n_estimators": 200,
#     "max_depth": 6,
#     "learning_rate": 0.1,
#     ...
# }

model = xgb.XGBClassifier(**model_params)
```

The `**` syntax is Python's dictionary unpacking operator. It takes every key-value pair in
the dictionary and passes them as keyword arguments:
```python
# These two lines are IDENTICAL:
model = xgb.XGBClassifier(**model_params)
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=50,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
```

The beauty: you can add/remove hyperparameters in params.yaml without touching Python code.

**Every hyperparameter explained:**

**`n_estimators: 200`** -- Number of boosting rounds (trees). Each tree corrects errors from
the previous ones. 200 is a moderate number. Too few (10) = underfitting. Too many (10000) =
slow training, risk of overfitting. With early stopping, training may stop before 200 if
validation performance plateaus.

**`max_depth: 6`** -- Maximum depth of each tree. A depth-6 tree can model interactions
between up to 6 features. Deeper trees capture more complex patterns but overfit more easily.
Default is 6. Range in practice: 3-10. For fraud detection with 29 features, 6 is reasonable.

**`learning_rate: 0.1`** -- Also called `eta`. Shrinks the contribution of each tree by this
factor. Lower values (0.01) need more trees but generalize better. Higher values (0.3) learn
faster but risk overshooting. 0.1 is the standard starting point. The tradeoff:
`learning_rate * n_estimators` should be roughly constant.

**`min_child_weight: 3`** -- Minimum sum of instance weights needed in a child node. For
binary classification with uniform weights, this effectively means "minimum 3 samples in a
leaf." Higher values make the model more conservative (fewer splits on rare patterns). Set
to 3 here to avoid splits on noise. With only 492 fraud cases, preventing overfitting to
individual fraud transactions is critical.

**`subsample: 0.8`** -- Fraction of training rows used per tree. Each tree sees a random 80%
of the training data. This is bagging (bootstrap aggregating) and reduces overfitting. The
remaining 20% acts as implicit regularization. Range: 0.5-1.0. Below 0.5 is too aggressive.

**`colsample_bytree: 0.8`** -- Fraction of features used per tree. Each tree only sees 80%
of features (randomly selected). This prevents any single feature from dominating and
increases diversity among trees. Combined with subsample, each tree sees 80% of rows AND
80% of columns -- substantial randomization.

**`scale_pos_weight: 50`** -- THIS IS THE CRITICAL PARAMETER for fraud detection. It tells
XGBoost that positive samples (fraud, Class=1) are 50x more important than negative samples
(legitimate, Class=0). The ideal value is approximately:
```
scale_pos_weight = count(negative) / count(positive)
                 = 284315 / 492
                 = 577.9
```
We use 50 instead of 578 because:
- 578 would make the model extremely aggressive -- it would flag nearly everything as fraud
  to avoid missing any real fraud, destroying precision.
- 50 is a moderate balance. It says "fraud is 50x more important" but still penalizes false
  positives enough to maintain useful precision.
- The optimal value depends on business cost: if missing one fraud costs $5000 but one false
  positive costs $5, the ratio is 1000:1. Real-world values are tuned empirically.

**`eval_metric: aucpr`** -- Area Under the Precision-Recall Curve. For imbalanced datasets,
AUC-PR is far superior to AUC-ROC because:
- AUC-ROC can be misleadingly high (0.99) even when the model catches very few fraud cases.
  This happens because true negative rate dominates.
- AUC-PR focuses on precision and recall for the MINORITY class, which is what we actually
  care about.

**`random_state: 42`** -- Same seed as the data split. Ensures reproducibility of the
internal randomness in XGBoost (subsample, colsample_bytree use random selection).

**`n_jobs: -1`** -- Use ALL available CPU cores for parallel training. XGBoost can parallelize
tree construction across cores. -1 means "detect the number of cores automatically."
On a 16-core machine, this gives ~10x speedup over single-threaded training.

### 1.4 The `training` Section

```yaml
training:
  experiment_name: fraud-detection
  cv_folds: 5
  early_stopping_rounds: 20
```

**`experiment_name: fraud-detection`** -- The MLflow experiment name. All runs logged during
training go under this experiment. In MLflow UI, you would see:
```
Experiments:
  [+] fraud-detection
      Run 1: 2024-01-15 14:32  recall=0.82  precision=0.91
      Run 2: 2024-01-15 15:10  recall=0.85  precision=0.88
      ...
```

**`cv_folds: 5`** -- Number of cross-validation folds. The training data is split into 5
equal parts. The model trains on 4 parts and validates on the 5th, rotating 5 times. This
gives 5 metric estimates, and the average is more reliable than a single train/val split.
5 folds is the standard choice. 10 folds is more expensive but reduces variance further.

**`early_stopping_rounds: 20`** -- If validation metric does not improve for 20 consecutive
boosting rounds, stop training. This prevents overfitting and saves time. Example:
```
Round 1:   val_aucpr = 0.70
Round 50:  val_aucpr = 0.85
Round 100: val_aucpr = 0.88
Round 120: val_aucpr = 0.88  (no improvement for 20 rounds -- STOP)
```
Without early stopping, the model would train all 200 rounds even if round 120 was optimal,
wasting compute and potentially overfitting.

### 1.5 The `thresholds` Section -- Quality Gates

```yaml
thresholds:
  min_recall: 0.02
  min_precision: 0.50
  min_f1: 0.03
  min_auc_roc: 0.90
```

These are MINIMUM acceptable metric values. If the model fails any threshold, the pipeline
can block deployment. This is a quality gate -- it prevents a bad model from reaching
production.

**`min_recall: 0.02`** -- The model must catch at least 2% of actual fraud cases. This is
intentionally low, likely a development/iteration threshold rather than production-ready.

**`min_precision: 0.50`** -- When the model flags a transaction as fraud, at least 50% of
those flags must actually be fraud. This prevents an "everything is fraud" model from passing.

**`min_f1: 0.03`** -- The harmonic mean of precision and recall. F1 = 2 * (precision * recall)
/ (precision + recall). This catches models that excel at one metric by destroying the other.

**`min_auc_roc: 0.90`** -- The model's discrimination ability must be at least 0.90. A random
model scores 0.50, a perfect model scores 1.00. 0.90 is a strong but achievable bar.

**Why different thresholds for different environments?** In a real production setup:

```yaml
# params_dev.yaml
thresholds:
  min_recall: 0.02     # Loose -- let experiments through quickly
  min_precision: 0.30
  min_auc_roc: 0.80

# params_staging.yaml
thresholds:
  min_recall: 0.70     # Stricter -- prove it works on staging data
  min_precision: 0.60
  min_auc_roc: 0.90

# params_prod.yaml
thresholds:
  min_recall: 0.85     # Strictest -- do not deploy garbage to production
  min_precision: 0.75
  min_auc_roc: 0.95
```

DVC supports this pattern: `dvc params diff` can compare parameter files, and CI/CD pipelines
can select the appropriate file for each environment.

### 1.6 The `monitoring` Section

```yaml
monitoring:
  drift_threshold: 0.05
  performance_window_days: 7
  alert_on_drift: true
```

**`drift_threshold: 0.05`** -- If the statistical distance between training data distribution
and incoming production data exceeds 0.05, the system flags data drift. This uses tests like
the Kolmogorov-Smirnov test or Population Stability Index. 0.05 is a conservative threshold --
even small shifts trigger alerts.

**`performance_window_days: 7`** -- The monitoring system looks at the last 7 days of
predictions when calculating performance metrics. This rolling window adapts to recent
trends. Too short (1 day) = noisy. Too long (90 days) = slow to detect degradation.

**`alert_on_drift: true`** -- When drift is detected, actively send alerts (email, Slack,
PagerDuty). Set to `false` during development to avoid alert fatigue.

### 1.7 The `aws` Section

```yaml
aws:
  region: us-east-1
  s3_bucket: mlops-fraud-detection-011015903780
  ecr_repository: mlops-fraud-detection
  lambda_function: mlops-fraud-prediction
  api_name: fraud-detection-api
```

**`region: us-east-1`** -- AWS region. us-east-1 (N. Virginia) is the most commonly used
region, has the most services available, and is often cheapest.

**`s3_bucket: mlops-fraud-detection-011015903780`** -- The S3 bucket for storing DVC-tracked
data, model artifacts, and pipeline outputs. The suffix `011015903780` is the AWS account ID,
ensuring global uniqueness (S3 bucket names must be globally unique across all AWS accounts).

**`ecr_repository: mlops-fraud-detection`** -- Amazon Elastic Container Registry repo where
Docker images are pushed. The inference container image is stored here.

**`lambda_function: mlops-fraud-prediction`** -- AWS Lambda function name for serverless
model inference. API Gateway routes requests to this function.

**`api_name: fraud-detection-api`** -- The API Gateway REST API name. This is the public-facing
endpoint that clients call to get fraud predictions.

### 1.8 Real-World Example: Different params.yaml Per Environment

A team can use DVC's parameter system to manage multiple environments:

```bash
# Developer working on a new feature branch:
cp params_dev.yaml params.yaml
dvc repro                          # Runs with loose thresholds, fast iteration

# Merging to staging:
cp params_staging.yaml params.yaml
dvc repro                          # Runs with stricter thresholds

# Deploying to production:
cp params_prod.yaml params.yaml
dvc repro                          # Runs with strictest thresholds
```

An even cleaner approach: environment variables + param overrides in CI/CD:
```bash
# In GitHub Actions:
dvc params modify thresholds.min_recall 0.85
dvc params modify thresholds.min_precision 0.75
dvc repro
```

---

## 2. dvc.yaml -- The ML Pipeline as Code

### 2.1 What Is a DVC Pipeline?

A DVC pipeline is a Directed Acyclic Graph (DAG) of stages. Each stage is a command that
takes inputs (dependencies) and produces outputs. DVC tracks the relationships and only
reruns stages whose inputs have changed.

**Key terms:**
- **Directed**: data flows in one direction (ingest -> validate -> preprocess -> train -> evaluate)
- **Acyclic**: no circular dependencies (evaluate cannot depend on ingest while ingest depends
  on evaluate)
- **Graph**: stages are nodes, dependencies are edges

### 2.2 The DAG Visualization

```
+----------+     +----------+     +------------+     +---------+     +----------+
|  ingest  | --> | validate | --> | preprocess | --> |  train  | --> | evaluate |
+----------+     +----------+     +------------+     +---------+     +----------+
     |                |                 |                  |               |
  Downloads        Checks           Splits data,       Trains         Tests on
  raw CSV          data quality      scales features    XGBoost       held-out
  from source      (schema,          into train/val/    model         test set,
                    nulls,           test sets                        generates
                    types)                                            metrics
```

You can generate this with:
```bash
$ dvc dag

    +--------+
    | ingest |
    +--------+
        *
        *
        *
    +----------+
    | validate |
    +----------+
        *
        *
        *
    +------------+
    | preprocess |
    +------------+
        *
        *
        *
      +-------+
      | train |
      +-------+
        *
        *
        *
    +----------+
    | evaluate |
    +----------+
```

### 2.3 Complete dvc.yaml with Annotations

```yaml
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps:
      - src/data/ingest.py
      - params.yaml
    outs:
      - data/raw/creditcard.csv:
          cache: true
    params:
      - data.raw_path

  validate:
    cmd: python -m src.data.validate
    deps:
      - src/data/validate.py
      - data/raw/creditcard.csv
    params:
      - data
      - features

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - src/data/validate.py
      - data/raw/creditcard.csv
    outs:
      - data/processed/X_train.csv:
          cache: true
      - data/processed/X_val.csv:
          cache: true
      - data/processed/X_test.csv:
          cache: true
      - data/processed/y_train.csv:
          cache: true
      - data/processed/y_val.csv:
          cache: true
      - data/processed/y_test.csv:
          cache: true
      - data/processed/scaler.pkl:
          cache: true
    params:
      - data
      - features

  train:
    cmd: python -m src.models.train
    deps:
      - src/models/train.py
      - data/processed/X_train.csv
      - data/processed/y_train.csv
      - data/processed/X_val.csv
      - data/processed/y_val.csv
    outs:
      - models/model.pkl:
          cache: true
    params:
      - model
      - training
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.models.evaluate
    deps:
      - src/models/evaluate.py
      - models/model.pkl
      - data/processed/X_test.csv
      - data/processed/y_test.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - metrics/confusion_matrix.json:
          cache: false
      - metrics/roc_curve.json:
          cache: false
```

### 2.4 DVC Stage Anatomy -- Every Field Explained

**`cmd:` -- The Command to Run**

```yaml
cmd: python -m src.data.ingest
```

This runs the `ingest.py` module inside `src/data/`. The `-m` flag tells Python to run a
module by name (using dotted package notation) rather than by file path. This matters because:
- It ensures Python resolves imports correctly relative to the project root.
- `python -m src.data.ingest` works regardless of your current directory.
- `python src/data/ingest.py` might fail with import errors if `src` is not on `sys.path`.

**`deps:` -- Dependencies (What Triggers a Rerun)**

```yaml
deps:
  - src/data/ingest.py    # If you change the script logic, stage reruns
  - params.yaml           # If any param changes, stage reruns
```

DVC computes an MD5 hash of every dependency. On `dvc repro`, it compares current hashes with
the hashes stored in `dvc.lock`. If ANY hash differs, the stage reruns. This is the core
intelligence of DVC pipelines.

Why is `src/data/ingest.py` a dependency? Because if you change the ingestion logic (say, you
add data cleaning), the output data will be different, so downstream stages need to rerun.

Why is `params.yaml` a dependency of ingest? Because it reads `data.raw_path` to know where
to save the file. If that path changes, the output location changes.

**`outs:` -- Outputs (What DVC Tracks and Versions)**

```yaml
outs:
  - data/raw/creditcard.csv:
      cache: true
```

DVC-tracked outputs are:
1. Added to `.gitignore` automatically (git does not track them; DVC does)
2. Stored in the DVC cache (`.dvc/cache/`) by content hash
3. Pushed to the DVC remote (S3) with `dvc push`

**`cache: true` vs `cache: false`**

- `cache: true` (default): DVC stores a copy in its local cache and can push to remote.
  Use for large binary files (datasets, models) that you want to version and share.
- `cache: false`: DVC tracks the file's hash for change detection but does NOT cache it.
  Use for small text files (metrics JSON) that are committed directly to git. There is no
  point caching a 233-byte JSON file when git handles text files perfectly.

Notice the pattern in this project:
- `data/*.csv`, `models/*.pkl`, `data/processed/scaler.pkl` use `cache: true` (large, binary)
- `metrics/*.json` uses `cache: false` (small, text, viewable with `dvc metrics show`)

**`params:` -- Parameter Dependencies**

```yaml
params:
  - data.raw_path       # Only reruns if data.raw_path changes
```

vs:

```yaml
params:
  - data                # Reruns if ANY field under data changes
  - features            # Reruns if ANY field under features changes
```

The first form (`data.raw_path`) is a specific parameter dependency. The stage only reruns
if that exact value changes.

The second form (`data`, `features`) is a section-level dependency. The stage reruns if ANY
field in those sections changes. This is safer but may cause unnecessary reruns.

**`metrics:` -- Special Metric Outputs**

```yaml
metrics:
  - metrics/train_metrics.json:
      cache: false
```

Metrics are outputs with a special property: DVC can display and compare them:

```bash
$ dvc metrics show
Path                        recall    precision    f1_score    auc_roc
metrics/train_metrics.json  0.8571    0.9231       0.8889      0.9712
metrics/eval_metrics.json   0.8367    0.9114       0.8724      0.9685

$ dvc metrics diff
Path                        Metric     HEAD      workspace
metrics/eval_metrics.json   recall     0.8200    0.8367
metrics/eval_metrics.json   precision  0.9050    0.9114
```

This is incredibly powerful for experiment tracking. You can compare metrics across git
branches, across commits, without opening MLflow.

**`plots:` -- Visualization Data**

```yaml
plots:
  - metrics/confusion_matrix.json:
      cache: false
  - metrics/roc_curve.json:
      cache: false
```

DVC can render these as interactive plots:
```bash
$ dvc plots show
# Opens a browser with interactive ROC curve and confusion matrix
```

### 2.5 Stage-by-Stage Deep Dive

**STAGE 1: ingest**

```yaml
ingest:
  cmd: python -m src.data.ingest
  deps:
    - src/data/ingest.py
    - params.yaml
  outs:
    - data/raw/creditcard.csv:
        cache: true
  params:
    - data.raw_path
```

Purpose: download the credit card fraud dataset and save it as a CSV. This is the entry point
of the entire pipeline.

Trigger conditions (any of these causes a rerun):
- `src/data/ingest.py` is modified (MD5 hash changes)
- `params.yaml` is modified
- `data.raw_path` value changes in params.yaml

Output: `data/raw/creditcard.csv` (~160MB, cached in DVC). This file is large, which is why
DVC tracks it instead of git. Git would bloat the repository with every version of this file.

**STAGE 2: validate**

```yaml
validate:
  cmd: python -m src.data.validate
  deps:
    - src/data/validate.py
    - data/raw/creditcard.csv
  params:
    - data
    - features
```

Purpose: check data quality -- schema validation, null checks, type checking, expected
column presence. This is a gate that prevents garbage data from flowing downstream.

Notice: this stage has NO `outs`. It is a pure validation step. If validation fails, the
script exits with a non-zero code, and `dvc repro` halts the entire pipeline. It depends on
the raw CSV (if the data changes, re-validate) and on `data` + `features` params (if
expected columns change, re-validate).

**STAGE 3: preprocess**

```yaml
preprocess:
  cmd: python -m src.data.preprocess
  deps:
    - src/data/preprocess.py
    - src/data/validate.py       # <-- interesting dependency
    - data/raw/creditcard.csv
  outs:
    - data/processed/X_train.csv    # Feature matrix for training
    - data/processed/X_val.csv      # Feature matrix for validation
    - data/processed/X_test.csv     # Feature matrix for testing
    - data/processed/y_train.csv    # Labels for training
    - data/processed/y_val.csv      # Labels for validation
    - data/processed/y_test.csv     # Labels for testing
    - data/processed/scaler.pkl     # Fitted StandardScaler object
  params:
    - data
    - features
```

Why does preprocess depend on `src/data/validate.py`? Because the preprocessing script likely
imports validation functions to validate data before processing. If the validation logic
changes, preprocessing should rerun to ensure the new checks are applied.

This stage produces 7 outputs:
- 6 CSV files: the train/val/test split of features (X) and labels (y)
- 1 pickle file: the fitted scaler, needed at inference time to scale new data the same way

The scaler is critical. In production, when a new transaction arrives, it must be scaled
using the SAME mean and std computed from training data. The scaler.pkl preserves these
statistics.

**STAGE 4: train**

```yaml
train:
  cmd: python -m src.models.train
  deps:
    - src/models/train.py
    - data/processed/X_train.csv
    - data/processed/y_train.csv
    - data/processed/X_val.csv
    - data/processed/y_val.csv
  outs:
    - models/model.pkl:
        cache: true
  params:
    - model
    - training
  metrics:
    - metrics/train_metrics.json:
        cache: false
```

Dependencies include training AND validation data. The validation data is used for early
stopping -- the model monitors validation performance each round and stops if it stops
improving.

Output: `models/model.pkl` (~630KB). The trained XGBoost model serialized with pickle/joblib.
This is cached by DVC because you want to version models -- "give me the model from last
Tuesday" is a real request in production teams.

Metrics: `metrics/train_metrics.json` with `cache: false`. This small JSON file is committed
to git directly. It contains training metrics (recall, precision, F1, AUC-ROC) from cross-
validation. The `cache: false` means DVC does not store this in its cache -- git handles it.

Param dependencies: `model` and `training` sections. Change ANY hyperparameter and this stage
reruns. Change `data.test_size`? Train does NOT rerun directly -- but preprocess reruns first
(because it depends on `data`), producing new training files, which triggers train to rerun
(because its data deps changed).

**STAGE 5: evaluate**

```yaml
evaluate:
  cmd: python -m src.models.evaluate
  deps:
    - src/models/evaluate.py
    - models/model.pkl
    - data/processed/X_test.csv
    - data/processed/y_test.csv
  metrics:
    - metrics/eval_metrics.json:
        cache: false
  plots:
    - metrics/confusion_matrix.json:
        cache: false
    - metrics/roc_curve.json:
        cache: false
```

This stage uses the TEST set -- data the model has never seen during training. It is the
final, unbiased assessment. Depends on the trained model and test data.

No `outs` field -- only `metrics` and `plots`. All three outputs are `cache: false` because
they are small JSON files that belong in git. These enable `dvc metrics show` and `dvc plots
show`.

### 2.6 How `dvc repro` Works: Smart Caching

```bash
$ dvc repro
```

DVC walks the DAG from top to bottom:
1. Check `ingest`: compare current hashes of deps/params with `dvc.lock`. If same, skip.
2. Check `validate`: same check. If ingest was skipped and validate's other deps are
   unchanged, skip.
3. Check `preprocess`: same check. Only run if something upstream changed.
4. Check `train`: same check.
5. Check `evaluate`: same check.

**Real example: changing a hyperparameter:**

```bash
# Edit params.yaml: change learning_rate from 0.1 to 0.05
$ dvc repro
```

What happens:
- `ingest`: deps unchanged, SKIPPED
- `validate`: deps unchanged, SKIPPED
- `preprocess`: deps unchanged (data/features sections untouched), SKIPPED
- `train`: params.model.params.learning_rate changed! RERUNNING
- `evaluate`: model.pkl changed (new model was trained), RERUNNING

Only 2 of 5 stages run. This saves enormous time when your ingest+preprocess takes 30 minutes
on a large dataset.

**Another example: changing test_size:**

```bash
# Edit params.yaml: change test_size from 0.2 to 0.3
$ dvc repro
```

- `ingest`: SKIPPED (data.raw_path unchanged)
- `validate`: RERUNNING (data section changed, it depends on `data`)
- `preprocess`: RERUNNING (data section changed, plus validate.py dep was "run")
- `train`: RERUNNING (X_train.csv changed because different split)
- `evaluate`: RERUNNING (model.pkl changed + X_test.csv changed)

4 of 5 stages run. Only ingest is cached.

---

## 3. dvc.lock -- The Reproducibility Receipt

### 3.1 What dvc.lock Stores

`dvc.lock` is an auto-generated file that records the exact state of the pipeline after the
last successful `dvc repro`. It contains:

For every stage:
- The command that was run
- MD5 hash and file size of every dependency
- The exact parameter values used
- MD5 hash and file size of every output

Example from the actual dvc.lock:

```yaml
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps:
    - path: params.yaml
      hash: md5
      md5: b1f7455c38ae5d84287ded992ce84b11
      size: 918
    - path: src/data/ingest.py
      hash: md5
      md5: 0638bfeab66f0ae59919942ea1312c18
      size: 2148
    params:
      params.yaml:
        data.raw_path: data/raw/creditcard.csv
    outs:
    - path: data/raw/creditcard.csv
      hash: md5
      md5: 9cdef7967af55b32a2f8a86fb08f5d7f
      size: 168171130
```

### 3.2 How DVC Uses Hashes

The MD5 hash `9cdef7967af55b32a2f8a86fb08f5d7f` is a 128-bit fingerprint of the file's
content. Two files with the same hash have identical content (with astronomically high
probability).

DVC's change detection algorithm:
```
1. Compute MD5 of current file
2. Compare with MD5 in dvc.lock
3. If different -> file has changed -> rerun this stage
4. If same -> file is unchanged -> potentially skip this stage
```

The file sizes are recorded too (e.g., `size: 168171130` = ~160MB). This is a quick pre-check:
if the size changed, the hash definitely changed, so DVC can skip the expensive hash
computation.

### 3.3 Why dvc.lock is Committed to Git

dvc.lock MUST be committed to git. It enables reproducibility:

```bash
# Six months from now, you need to reproduce the exact results from commit abc123:
$ git checkout abc123
$ dvc checkout          # Restores data files to match dvc.lock from that commit
$ dvc repro             # Verifies everything matches; all stages SKIP if data is cached
```

Without dvc.lock in git, you could not reconstruct which version of the data, code, and
parameters produced a given model. The combination of `git log` (code/config history) and
`dvc.lock` (data/model hashes) gives you complete experiment lineage.

### 3.4 Key Observations from Our dvc.lock

Looking at the actual file:
- `data/raw/creditcard.csv` is 168,171,130 bytes (~160MB). This confirms why git cannot track it.
- `models/model.pkl` is 644,882 bytes (~630KB). Small for a model, but still binary.
- `metrics/eval_metrics.json` is only 233 bytes. Definitely belongs in git, not DVC cache.
- `data/processed/X_train.csv` is 111,692,645 bytes (~106MB). The largest processed file.
- `data/processed/scaler.pkl` is only 927 bytes. The scaler just stores mean and std for each
  feature -- very compact.

---

## 4. DVC Remote Configuration (.dvc/config)

```
[core]
    remote = s3remote

['remote "s3remote"']
    url = s3://mlops-fraud-detection-011015903780/dvc-store
    region = us-east-1
```

### 4.1 `[core] remote = s3remote`

This sets the default remote for `dvc push` and `dvc pull`. Without this, you would need to
specify `--remote s3remote` on every command. You can have multiple remotes:

```
[core]
    remote = s3remote

['remote "s3remote"']
    url = s3://my-bucket/dvc-store

['remote "gcs-backup"']
    url = gs://my-backup-bucket/dvc-store
```

### 4.2 The S3 Remote URL

`s3://mlops-fraud-detection-011015903780/dvc-store`

- `s3://` -- protocol prefix for Amazon S3
- `mlops-fraud-detection-011015903780` -- the bucket name (matches params.yaml)
- `/dvc-store` -- a prefix (folder) within the bucket

### 4.3 How DVC Stores Files in S3 (Content-Addressable Storage)

DVC does NOT store files with their original names. It uses content-addressable storage:

```
# Local file:
data/raw/creditcard.csv  (MD5: 9cdef7967af55b32a2f8a86fb08f5d7f)

# Stored in S3 as:
s3://mlops-fraud-detection-011015903780/dvc-store/9c/def7967af55b32a2f8a86fb08f5d7f
```

The MD5 hash becomes the path: first two characters as a directory (`9c/`), rest as filename.
This is the same strategy git uses for its object store.

Benefits:
- **Deduplication**: if two files have the same content, they share one copy in S3.
- **Integrity**: the path IS the hash, so corruption is immediately detectable.
- **No naming conflicts**: different branches can have different files at `data/raw/data.csv`
  without collision -- they are stored by content hash, not by name.

### 4.4 `dvc push` and `dvc pull`

```bash
# After training, push all DVC-tracked files to S3:
$ dvc push
Pushing to s3://mlops-fraud-detection-011015903780/dvc-store
  8 files pushed

# On another machine or by another team member:
$ git clone https://github.com/your-org/mlops-fraud-detection.git
$ cd mlops-fraud-detection
$ dvc pull
Fetching from s3://mlops-fraud-detection-011015903780/dvc-store
  8 files fetched
  8 files checked out
```

### 4.5 Real Example: New Team Member Onboarding

```bash
# Day 1: new data scientist joins the team
$ git clone git@github.com:company/mlops-fraud-detection.git
$ cd mlops-fraud-detection

# At this point: dvc.yaml, dvc.lock, params.yaml are present (from git)
# But data/ and models/ directories are EMPTY (not in git, managed by DVC)

$ pip install -r requirements.txt       # Installs DVC among other things
$ aws configure                          # Set up AWS credentials for S3 access

$ dvc pull
# DVC reads dvc.lock, finds all file hashes, downloads them from S3
# data/raw/creditcard.csv:        160MB downloaded
# data/processed/X_train.csv:     106MB downloaded
# data/processed/X_val.csv:        15MB downloaded
# data/processed/X_test.csv:       30MB downloaded
# ...
# models/model.pkl:               630KB downloaded

$ dvc repro
# ALL stages SKIP -- everything matches dvc.lock
# The new team member has an exact replica of the project state

$ dvc metrics show
# See the exact same metrics as everyone else on the team
```

---

## 5. requirements.txt -- Every Dependency Explained

```
# ML & Data
scikit-learn>=1.3.0
xgboost>=2.0.0
pandas>=2.1.0
numpy>=1.24.0
imbalanced-learn>=0.11.0

# Experiment Tracking
mlflow>=2.9.0

# Data Versioning
dvc[s3]>=3.30.0

# Data Validation
great-expectations>=0.18.0
pandera>=0.17.0

# Model Serving
fastapi>=0.104.0
uvicorn>=0.24.0
mangum>=0.17.0

# Monitoring
evidently>=0.4.0

# AWS
boto3>=1.29.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
httpx>=0.25.0

# Code Quality
ruff>=0.1.0
black>=23.0.0
mypy>=1.7.0
pre-commit>=3.5.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.0
joblib>=1.3.0
requests>=2.31.0
```

### 5.1 Grouped by Purpose

**ML and Data (core computation):**
- `scikit-learn` -- train_test_split, StandardScaler, classification_report, metrics. The
  Swiss army knife of ML.
- `xgboost` -- the gradient boosting model. The actual fraud detection engine.
- `pandas` -- DataFrame operations. Loading CSVs, feature manipulation, data exploration.
- `numpy` -- numerical arrays underlying all computation. pandas and scikit-learn depend on it.
- `imbalanced-learn` -- SMOTE and other techniques for handling class imbalance. May be used
  for oversampling the minority (fraud) class.

**Experiment Tracking:**
- `mlflow` -- logs parameters, metrics, and artifacts for every training run. Provides a UI
  for comparing experiments.

**Data Versioning:**
- `dvc[s3]` -- DVC with S3 support. The `[s3]` extra installs `boto3` and `s3fs` as
  additional dependencies for S3 remote storage.

**Data Validation:**
- `great-expectations` -- declarative data quality checks. "Column 'Amount' should have no
  nulls and be non-negative."
- `pandera` -- schema validation for pandas DataFrames. Lightweight alternative/complement
  to Great Expectations.

**Model Serving:**
- `fastapi` -- high-performance async web framework for the prediction API.
- `uvicorn` -- ASGI server that runs FastAPI. The actual process that listens on a port.
- `mangum` -- adapter that wraps FastAPI for AWS Lambda. Lambda sends API Gateway events;
  Mangum translates them into ASGI requests that FastAPI understands.

**Monitoring:**
- `evidently` -- data drift detection, model performance monitoring, generates drift reports.

**AWS:**
- `boto3` -- AWS SDK for Python. Used for S3, Lambda, ECR, and any AWS service interaction.

**Testing:**
- `pytest` -- test runner. Discovers and runs all test files matching `test_*.py`.
- `pytest-cov` -- coverage plugin. Measures which lines of code are exercised by tests.
- `httpx` -- async HTTP client for testing FastAPI endpoints. FastAPI's TestClient uses httpx.

**Code Quality:**
- `ruff` -- extremely fast Python linter (written in Rust). Replaces flake8, isort, and more.
- `black` -- opinionated code formatter. "Any customer can have a car painted any color that
  he wants, so long as it is black." Zero configuration debates.
- `mypy` -- static type checker. Catches type errors before runtime.
- `pre-commit` -- git hook manager. Runs ruff, black, mypy automatically before each commit.

**Utilities:**
- `python-dotenv` -- loads `.env` files into environment variables. Keeps secrets out of code.
- `pyyaml` -- YAML parser for reading params.yaml and other config files.
- `joblib` -- efficient serialization for numpy arrays and sklearn objects. Used for
  saving/loading the scaler and potentially the model.
- `requests` -- HTTP library for API calls (data download, external services).

### 5.2 Version Pinning: `>=` vs `==`

This project uses `>=` (minimum version):
```
scikit-learn>=1.3.0    # Any version 1.3.0 or newer
```

Alternative strategies:

```
scikit-learn==1.3.0    # EXACT version. Maximum reproducibility but blocks updates.
scikit-learn>=1.3.0,<2.0.0  # Range. Allows patches and minor versions, blocks major.
scikit-learn~=1.3.0    # Compatible release. Same as >=1.3.0,<1.4.0.
```

| Strategy | Pros | Cons |
|----------|------|------|
| `>=` | Flexible, gets security patches | May break with incompatible updates |
| `==` | Perfectly reproducible | Misses security patches, dependency hell |
| `>=,<` | Balanced | More maintenance, manual upper bound |
| Lock file | Best of both worlds | Requires pip-tools, Poetry, or similar |

The best practice in production is to use `>=` in requirements.txt (the "abstract"
requirements) and a lock file like `requirements.lock` or `poetry.lock` (the "concrete"
resolved versions). This project uses the simpler `>=` approach.

### 5.3 Dependency Conflicts

Real example of a conflict:
```
# mlflow 2.9.0 requires sqlalchemy<2.1,>=1.4.0
# great-expectations 0.18.0 requires sqlalchemy>=1.3.2,<2.0
# These overlap only at sqlalchemy>=1.4.0,<2.0 -- pip must find a version in that range
```

How to diagnose:
```bash
$ pip install -r requirements.txt
# ERROR: Cannot install mlflow and great-expectations because these package
# versions have conflicting dependencies.

$ pip install pipdeptree
$ pipdeptree --warn fail    # Shows dependency tree and conflicts
```

Resolution strategies:
1. Relax version constraints (use `>=` instead of `==`)
2. Upgrade/downgrade the conflicting package
3. Use a virtual environment per project (always do this)
4. Use Poetry or pip-tools for dependency resolution

### 5.4 `pip freeze` vs Manually Maintained

```bash
$ pip freeze > requirements.txt
# Output includes EVERY installed package (200+ lines)
# Includes transitive dependencies you did not ask for
# Hard to read, hard to maintain

# vs manually maintained (what this project does)
# Only lists direct dependencies (25 packages)
# Grouped and commented for readability
# Transitive deps are resolved by pip automatically
```

The manual approach is better for open-source and learning projects. For production, use both:
a human-readable `requirements.in` and a machine-generated `requirements.txt` via `pip-compile`.

---

## 6. pyproject.toml -- Project Configuration

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mlops-fraud-detection"
version = "1.0.0"
description = "Production MLOps pipeline for credit card fraud detection"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["N806", "N803"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "model: Model quality tests",
]

[tool.mypy]
python_version = "3.10"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
disable_error_code = ["import-untyped"]
```

### 6.1 `[build-system]` -- How to Build This Package

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"
```

This tells tools like `pip` and `build` how to create a distributable package:
- `setuptools` -- the classic Python build tool. Generates `.whl` and `.tar.gz` files.
- `wheel` -- the binary distribution format. Faster to install than source distributions.
- `build-backend = "setuptools.build_meta"` -- the entry point PEP 517 uses to invoke the
  build. This is the standard for setuptools projects.

You need this if you ever want to `pip install -e .` (editable/development install) or
distribute the project as a package.

### 6.2 `[project]` -- Package Metadata

```toml
[project]
name = "mlops-fraud-detection"
version = "1.0.0"
description = "Production MLOps pipeline for credit card fraud detection"
requires-python = ">=3.10"
license = {text = "MIT"}
```

- `name` -- the installable package name. After `pip install -e .`, you can `import mlops_fraud_detection`.
- `version` -- follows semantic versioning (MAJOR.MINOR.PATCH). 1.0.0 means first stable release.
- `requires-python = ">=3.10"` -- enforces Python 3.10 minimum. Features used include:
  - Match statements (3.10+)
  - Improved type hints (3.10+)
  - Performance improvements in dictionary operations
- `license = {text = "MIT"}` -- permissive license. Allows commercial use.

### 6.3 `[tool.ruff]` -- The Fast Linter

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["N806", "N803"]
```

Ruff is a Python linter written in Rust, 10-100x faster than flake8. It replaces multiple
tools: flake8, isort, pyupgrade, and more.

**`target-version = "py310"`** -- tells ruff to check for Python 3.10 compatibility. It will
flag syntax that requires Python 3.11+ and suggest upgrades for patterns that have cleaner
syntax in 3.10.

**`line-length = 100`** -- maximum line length. The Python standard (PEP 8) says 79 characters.
Most modern projects use 88 (black's default) or 100. 100 is practical for modern wide screens.

**Rule codes explained:**

| Code | Source | What It Checks | Example |
|------|--------|---------------|---------|
| `E` | pycodestyle (errors) | PEP 8 style errors | `E501`: line too long |
| `F` | pyflakes | Logical errors | `F841`: unused variable, `F401`: unused import |
| `I` | isort | Import sorting | `I001`: unsorted imports |
| `N` | pep8-naming | Naming conventions | `N801`: class not CapWords, `N806`: variable not lowercase |
| `W` | pycodestyle (warnings) | PEP 8 style warnings | `W291`: trailing whitespace |
| `UP` | pyupgrade | Python version upgrades | `UP035`: use `from typing import X` instead of deprecated form |

**Why ignore N806 and N803?**

```python
# N806: Variable in function should be lowercase
# N803: Argument name should be lowercase

# In ML, this is STANDARD convention:
X_train = ...    # N806 violation! ruff says use x_train
y_train = ...    # OK (lowercase)

def train_model(X, y):    # N803 violation! ruff says use x
    ...
```

The ML community uses uppercase `X` for feature matrices (following mathematical convention
where matrices are uppercase) and lowercase `y` for label vectors. This conflicts with
Python's naming convention (PEP 8 says variables should be lowercase). Ignoring N806 and N803
lets ML code follow its own convention without linter noise.

### 6.4 `[tool.black]` -- The Code Formatter

```toml
[tool.black]
line-length = 100
target-version = ["py310"]
```

Black is the "uncompromising" code formatter. There is almost nothing to configure -- that is
the point. It eliminates all style debates by enforcing a single style.

`line-length = 100` -- must match ruff's setting. If black formats to 100 characters but ruff
complains at 80, they fight endlessly.

`target-version = ["py310"]` -- list format (not string like ruff). Tells black what syntax
to use. For example, it will use `X | Y` union types instead of `Union[X, Y]` because 3.10
supports the pipe syntax.

### 6.5 `[tool.pytest.ini_options]` -- Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "model: Model quality tests",
]
```

**`testpaths = ["tests"]`** -- pytest only looks in the `tests/` directory. Without this,
pytest scans the entire project, which could find test-like files in data directories or
virtual environments.

**`addopts = "-v --tb=short"`** -- default options added to every pytest run:
- `-v` (verbose): shows each test name and result instead of just dots
- `--tb=short`: on failure, shows a shortened traceback (just the failing line and assertion)
  instead of the full traceback

```bash
# Without -v:
....F..
# With -v:
tests/test_model.py::test_model_recall PASSED
tests/test_model.py::test_model_precision PASSED
tests/test_model.py::test_model_f1 FAILED     # <-- immediately visible
tests/test_api.py::test_health_endpoint PASSED
```

**`markers`** -- custom test categories. These let you run subsets:

```bash
$ pytest -m unit              # Only fast unit tests (seconds)
$ pytest -m integration       # Only integration tests (minutes, need services running)
$ pytest -m model             # Only model quality tests (need trained model)
$ pytest -m "not integration" # Everything except slow integration tests
```

This is critical in CI/CD. Pull request checks run `pytest -m unit` (fast feedback). Nightly
builds run `pytest -m "unit or integration or model"` (thorough but slow).

### 6.6 `[tool.mypy]` -- Static Type Checker

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
disable_error_code = ["import-untyped"]
```

**`python_version = "3.10"`** -- mypy checks type annotations assuming Python 3.10 semantics.

**`warn_return_any = false`** -- do NOT warn when a function returns `Any`. In ML code,
many libraries (pandas, numpy) return `Any` types because their return types are highly
dynamic. Enabling this would produce hundreds of false positives.

**`warn_unused_configs = true`** -- warn if a mypy config option has no effect. Helps catch
typos in the config itself.

**`ignore_missing_imports = true`** -- do NOT error when importing a package that has no type
stubs. Many ML libraries (xgboost, evidently, great-expectations) do not ship type stubs. Without
this, mypy would error on every `import xgboost` line.

**`disable_error_code = ["import-untyped"]`** -- suppress the specific error about importing
untyped packages. Belt-and-suspenders with `ignore_missing_imports` for thorough suppression.

### 6.7 Real Example: Adding a New Tool Config

If you wanted to add `isort` (import sorter) configuration:

```toml
# Add this to pyproject.toml:
[tool.isort]
profile = "black"           # Format imports to match black's style
known_first_party = ["src"] # Treat src as a local package
line_length = 100           # Match black and ruff
```

However, since ruff already handles import sorting (rule code `I`), adding isort would be
redundant. This is one of ruff's advantages -- it consolidates many tools.

---

## 7. .gitignore -- What We Track vs What We Do Not

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/
mlartifacts/

# DVC
/data/raw/*.csv
/data/processed/*.csv
/data/processed/*.pkl

# Models
models/*.pkl
models/*.joblib
*.model

# AWS
.aws/

# Environment
.env
.env.*
AWS_Credential

# OS
.DS_Store
Thumbs.db

# Terraform
infrastructure/terraform/.terraform/
infrastructure/terraform/*.tfstate
infrastructure/terraform/*.tfstate.backup
infrastructure/terraform/.terraform.lock.hcl

# Docker
*.tar

# Logs
*.log
logs/
.playwright-mcp/
```

### 7.1 Python Artifacts

```gitignore
__pycache__/      # Compiled bytecode cache directories
*.py[cod]         # .pyc, .pyo, .pyd files (compiled Python)
*$py.class        # Jython compiled files
*.so              # Compiled C extensions (Linux/Mac)
*.egg-info/       # Package metadata from setuptools
dist/             # Built distributions (pip install output)
build/            # Build artifacts
.eggs/            # Downloaded eggs during build
```

Why exclude these? They are generated artifacts. Every developer's machine creates different
bytecode (depends on Python version, OS, architecture). Committing them causes merge
conflicts and bloats the repo for zero benefit.

The `*.py[cod]` pattern is a character class: matches `.pyc` OR `.pyo` OR `.pyd`. This is
more concise than three separate lines.

### 7.2 Virtual Environments

```gitignore
.venv/
venv/
env/
```

Virtual environments contain hundreds of installed packages (often 500MB+). They are machine-
specific (contain compiled C extensions for your OS/architecture). The `requirements.txt` file
IS the portable representation of the environment. Never commit the venv itself.

Three directory names are covered because different conventions exist:
- `.venv/` -- recommended by Python documentation (hidden directory)
- `venv/` -- common convention
- `env/` -- alternative convention

### 7.3 IDE Files

```gitignore
.vscode/          # VS Code workspace settings, extensions, debug configs
.idea/            # JetBrains (PyCharm) project settings
*.swp             # Vim swap files (in-progress edits)
*.swo             # Vim swap file overflow
```

IDE settings are personal preference. One developer uses PyCharm with vim keybindings, another
uses VS Code with a custom theme. Committing these creates constant merge conflicts and
forces your editor preferences on others.

Exception: some teams DO commit `.vscode/settings.json` with project-specific settings
(Python interpreter path, linting rules). This is a team decision.

### 7.4 MLflow Data

```gitignore
mlruns/           # MLflow experiment data (local file store)
mlartifacts/      # MLflow artifact store (models, plots, etc.)
```

MLflow stores experiment data locally in `mlruns/`. This can grow to gigabytes with many
experiments. It is regenerable by rerunning the pipeline. In production, MLflow data lives
on a remote tracking server (not local files), making these directories irrelevant.

Why not track MLflow data in git?
- Binary files (model artifacts)
- Large and growing
- Regenerable from code + data
- In production, this lives on an MLflow server, not in the repo

### 7.5 DVC-Tracked Data Files

```gitignore
/data/raw/*.csv
/data/processed/*.csv
/data/processed/*.pkl
```

THIS IS THE MOST IMPORTANT SECTION. These files are managed by DVC, not git. DVC creates
`.dvc` pointer files (tiny text files with hash references) that ARE tracked by git. The
actual large files are stored in the DVC cache and pushed to S3.

The leading `/` makes these patterns relative to the repo root. Without `/`, they would
match `any/nested/data/raw/*.csv` too.

The relationship:
```
git tracks:       dvc.yaml, dvc.lock, params.yaml     (small text files)
DVC tracks:       data/raw/*.csv, data/processed/*     (large data files)
S3 stores:        actual file content (by hash)         (remote backup)
```

### 7.6 Model Files

```gitignore
models/*.pkl       # Pickled scikit-learn/XGBoost models
models/*.joblib    # Joblib-serialized models
*.model            # Generic model file extension
```

Same reasoning as data files. Models are produced by the training stage and tracked by DVC.
A 630KB model file seems small, but over hundreds of experiments, it adds up. More importantly,
model files are BINARY -- git cannot diff them meaningfully.

### 7.7 Environment and Credential Files

```gitignore
.env               # Environment variables (API keys, database URLs)
.env.*             # Environment-specific files (.env.prod, .env.staging)
AWS_Credential     # AWS access key and secret key
```

THIS IS A SECURITY-CRITICAL SECTION. These files contain secrets:

```bash
# .env (example)
AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
MLFLOW_TRACKING_URI=http://mlflow-server.internal:5000
DATABASE_URL=postgresql://user:password@host:5432/db
```

If these are committed to git:
1. They appear in git history FOREVER (even after deletion)
2. Anyone who clones the repo gets your AWS credentials
3. Automated bots scan GitHub for exposed credentials (they WILL find them)
4. Your AWS account gets compromised within hours

The `AWS_Credential` file is a project-specific credential file. The generic `.aws/` directory
is also excluded as it may contain AWS configuration with embedded credentials.

### 7.8 Terraform State Files

```gitignore
infrastructure/terraform/.terraform/      # Downloaded provider plugins (large, OS-specific)
infrastructure/terraform/*.tfstate        # Current infrastructure state
infrastructure/terraform/*.tfstate.backup # Previous state backup
infrastructure/terraform/.terraform.lock.hcl  # Provider version lock
```

Terraform state files deserve special attention:

**`*.tfstate`** -- contains the FULL state of your cloud infrastructure, including:
- Resource IDs
- IP addresses
- IAM role ARNs
- Database connection strings
- Sometimes PASSWORDS in plaintext

This is extremely sensitive. In production, terraform state is stored in a remote backend
(S3 + DynamoDB for locking) and NEVER committed to git.

**`.terraform/`** -- downloaded provider binaries. These are hundreds of megabytes of
platform-specific binaries. Regenerated by `terraform init`.

**`.terraform.lock.hcl`** -- actually, many teams DO commit this file. It locks provider
versions for reproducibility (similar to `package-lock.json` in Node.js). This project
excludes it, which means different developers might get different provider versions.

### 7.9 Other Exclusions

```gitignore
.DS_Store          # macOS folder metadata (hidden file created by Finder)
Thumbs.db          # Windows thumbnail cache
*.tar              # Docker image tarballs (large binary files)
*.log              # Log files (regenerable, potentially large)
logs/              # Log directory
.playwright-mcp/   # Playwright (browser automation) cache
```

`.DS_Store` is the most common accidental commit on macOS. It appears in every directory you
open in Finder. It contains folder view settings (icon size, sort order) -- completely
irrelevant to the project.

---

## Summary: How Everything Connects

The configuration files form a coherent system:

```
params.yaml          -- WHAT to build (data splits, model config, thresholds)
    |
    v
dvc.yaml             -- HOW to build it (pipeline stages, deps, outputs)
    |
    v
dvc.lock             -- PROOF it was built (hashes of everything used and produced)
    |
    v
.dvc/config          -- WHERE to store data (S3 remote configuration)
    |
    v
requirements.txt     -- WITH WHAT tools (Python dependencies)
    |
    v
pyproject.toml       -- FOLLOWING WHAT rules (linting, formatting, testing config)
    |
    v
.gitignore           -- PROTECTING WHAT (secrets, large files, generated artifacts)
```

When a data scientist changes `learning_rate` in params.yaml:
1. DVC detects the param change (dvc.yaml says `train` depends on `model` params)
2. `dvc repro` reruns `train` and `evaluate` stages (smart caching skips unchanged stages)
3. New metrics appear in `metrics/train_metrics.json` and `metrics/eval_metrics.json`
4. `dvc metrics diff` shows exactly what changed
5. MLflow logs the run with all parameters and metrics
6. The quality gate checks thresholds -- if the model passes, it can be deployed
7. `dvc push` uploads the new model and data to S3
8. `.gitignore` ensures secrets and large files stay out of git

This is the MLOps feedback loop: configure, run, measure, compare, iterate. The configuration
files are the control surface. The pipeline is the engine. DVC is the version control layer
that makes it all reproducible.
