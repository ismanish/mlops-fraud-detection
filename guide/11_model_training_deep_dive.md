# Model Training Deep Dive

This guide dissects every line of `src/models/train.py` and `src/models/evaluate.py`,
explaining every hyperparameter, every metric, every MLflow call, and every design decision.
When you finish this guide, you will understand not just *what* the code does, but *why*
every choice was made and what would happen if you changed it.

---

## 1. XGBoost Classifier — Every Hyperparameter Explained

### 1.1 What is Gradient Boosting?

Gradient boosting builds an **ensemble of weak learners** (small decision trees) where each
new tree tries to correct the mistakes made by all previous trees combined.

Think of it like a team of students taking a test:
- Student 1 answers all questions, gets 60% right
- Student 2 only looks at the questions Student 1 got wrong, and focuses there
- Student 3 only looks at the questions Students 1+2 still get wrong
- After 200 students, the team gets 98% right

Here is a concrete 3-tree example for fraud detection:

```
TREE 1: "Is the amount > $5000?"
         ├── YES → predict fraud (probability 0.6)
         └── NO  → predict legit (probability 0.3)

Residual errors after Tree 1:
  Transaction A: actual=fraud,  predicted=0.6  → residual = 0.4  (under-predicted)
  Transaction B: actual=legit,  predicted=0.3  → residual = -0.3 (over-predicted)
  Transaction C: actual=fraud,  predicted=0.3  → residual = 0.7  (badly under-predicted)

TREE 2: Trained on those residuals. Focuses on Transaction C.
         "Is transaction from a new device AND amount > $1000?"
         ├── YES → +0.5  (boost the fraud prediction)
         └── NO  → -0.1  (slight correction toward legit)

Combined after Tree 2:
  Transaction C: 0.3 + 0.5 = 0.8  (much better!)

TREE 3: Trained on remaining residuals after Trees 1+2.
         "Is transaction at 3AM AND in a different country than usual?"
         ├── YES → +0.3
         └── NO  → -0.05

Final prediction = Tree1 + learning_rate * Tree2 + learning_rate * Tree3
```

The key insight: each tree is **small and weak** on its own. The power comes from
**hundreds of trees**, each fixing what the previous ones missed. The `learning_rate`
controls how much each new tree's correction is trusted.

### 1.2 The Hyperparameters in Our params.yaml

From `params.yaml`:

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

And in `train.py`, line 59, they are unpacked into the XGBClassifier:

```python
model = xgb.XGBClassifier(**model_params)
```

The `**` operator unpacks the dictionary as keyword arguments, so this is equivalent to:

```python
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

Let us go through each one.

---

### 1.3 `n_estimators=200` — Number of Boosting Rounds

**What it does:** The total number of trees to build sequentially. Each tree tries to
correct errors from all previous trees.

**Too few (e.g., 10):**
```
Trees:  [T1] [T2] [T3] ... [T10]
                                  ↑ stops here
Error:  ████████████████████░░░░░░░░░░░░░░░░░░
        Still high! Model underfits.
        "I only had 10 chances to learn — I missed a lot of patterns."
```

**Just right (e.g., 200):**
```
Trees:  [T1] [T2] [T3] ... [T100] ... [T200]
                                             ↑ stops here
Error:  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Low! Model has captured most patterns.
```

**Too many (e.g., 10,000):**
```
Trees:  [T1] [T2] ... [T200] ... [T5000] ... [T10000]
                                                      ↑ stops here
Error on train:  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Error on test:   █████████████░░░░░░░░░░░░░░░░░░░░░░░
                 Overfitting! Model memorizes training noise.
                 Training error near zero, but test error goes back up.
```

**Why 200?** It is a sweet spot for this dataset. With `learning_rate=0.1`, each tree
makes a small correction, and 200 trees are enough to converge without overfitting.
The general rule: lower learning rate needs more estimators, higher learning rate
needs fewer.

**Interview insight:** In production, you would use early stopping (configured in
`params.yaml` as `early_stopping_rounds: 20`) to automatically find the best number.
Early stopping watches validation performance and stops training when it has not
improved for 20 rounds.

---

### 1.4 `max_depth=6` — Tree Depth

**What it does:** The maximum number of levels in each decision tree. Deeper trees can
capture more complex patterns but are more prone to overfitting.

**Visualize a depth-3 tree:**

```
Depth 0 (root):          [amount > $2000?]
                          /              \
Depth 1:        [hour > 22?]        [country != home?]
                 /        \           /            \
Depth 2:   [new_device?]  [Legit]  [amount>$500?]  [Legit]
            /       \                /         \
Depth 3: [Fraud]  [Legit]     [Fraud]      [Legit]
```

At each depth, the tree can make one more decision. So:
- Depth 1: 2 possible outcomes (1 split)
- Depth 2: 4 possible outcomes (can distinguish 4 patterns)
- Depth 3: 8 possible outcomes
- Depth 6: 64 possible outcomes (2^6)
- Depth 10: 1024 possible outcomes

**Depth 6 vs. lower/higher:**

| max_depth | Patterns Captured | Overfitting Risk | Training Speed |
|-----------|-------------------|------------------|----------------|
| 2         | Very simple       | Very low         | Very fast      |
| 4         | Moderate          | Low              | Fast           |
| 6         | Complex           | Moderate         | Moderate       |
| 10        | Very complex      | High             | Slow           |
| 15        | Nearly any        | Very high        | Very slow      |

**Why 6?** Fraud detection requires moderately complex patterns. A depth-2 tree might
only learn "high amount = fraud," but a depth-6 tree can learn interactions like
"high amount AND nighttime AND new device AND foreign country AND no prior history
AND unusual merchant category." Six-way interactions are usually enough for tabular data.

---

### 1.5 `learning_rate=0.1` — Shrinkage

**What it does:** Multiplies each tree's contribution by this factor. Also called `eta`
in XGBoost documentation.

**The math:**

Without shrinkage (learning_rate = 1.0):
```
prediction = Tree1 + Tree2 + Tree3 + ... + Tree200
```

With shrinkage (learning_rate = 0.1):
```
prediction = Tree1 + 0.1 * Tree2 + 0.1 * Tree3 + ... + 0.1 * Tree200
```

More precisely, the prediction after `t` trees is:

```
F_t(x) = F_{t-1}(x) + learning_rate * h_t(x)

where:
  F_t(x)     = combined prediction after t trees
  F_{t-1}(x) = combined prediction after t-1 trees
  h_t(x)     = the t-th tree's raw prediction
```

**Why shrinkage helps:**

Think of it as walking toward a target:
- `learning_rate=1.0`: Take full-sized steps. Might overshoot and oscillate.
- `learning_rate=0.1`: Take baby steps. Takes longer but arrives more precisely.
- `learning_rate=0.01`: Tiny steps. Very precise but needs thousands of trees.

**The tradeoff with `n_estimators`:**

```
learning_rate=1.0  + n_estimators=20   → Fast but overfit, unstable
learning_rate=0.1  + n_estimators=200  → Good balance (our choice)
learning_rate=0.01 + n_estimators=2000 → Slightly better but 10x slower
```

The product `learning_rate * n_estimators` roughly determines model capacity.
For our project: 0.1 * 200 = 20 "effective trees." This is a well-known heuristic:
you want this product to be roughly 10-50 for most problems.

---

### 1.6 `min_child_weight=3` — Minimum Sum of Instance Weight

**What it does:** The minimum sum of instance weight (hessian) needed in a child node.
If a split would create a child node where the sum of weights is less than 3, that
split is not made.

**In plain English:** Do not create a leaf node that covers fewer than ~3 data points
(for unweighted data, min_child_weight approximates minimum samples per leaf).

**How it prunes:**

```
Consider this split:
                [amount > $9999?]
                /              \
        [3000 samples]    [2 samples]
        (weight = 3000)   (weight = 2)

With min_child_weight=3:
  Right child has weight 2 < 3 → SPLIT REJECTED
  The node stays as a leaf instead of splitting further.

Why? A leaf with only 2 samples is probably noise.
Those 2 transactions that happened to share a pattern are not
a reliable signal — they could be coincidence.
```

**Impact on fraud detection:** Since fraud is rare (~0.17% of transactions), without
this constraint, trees might create tiny leaf nodes containing 1-2 fraud cases,
learning noise rather than real patterns. Setting it to 3 ensures each decision
is backed by at least a small group of examples.

---

### 1.7 `subsample=0.8` — Row Sampling

**What it does:** Each tree is trained on a random 80% of the training data (sampled
without replacement).

**Why randomness helps (the bagging concept):**

```
Full dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  (10 samples)

Tree 1 sees:  [1, 2, 4, 5, 6, 7, 8, 10]         (80% random sample)
Tree 2 sees:  [1, 3, 4, 5, 7, 8, 9, 10]         (different 80%)
Tree 3 sees:  [2, 3, 4, 5, 6, 8, 9, 10]         (different 80%)
```

Each tree sees a slightly different view of the data, so they make different
mistakes. When you combine them, the individual mistakes cancel out (like how
polling multiple people gives a better answer than asking one expert).

This is called **bagging** (bootstrap aggregating). It reduces variance (overfitting)
without increasing bias (underfitting).

**Why 0.8 and not 0.5 or 1.0?**
- `1.0`: No sampling. Each tree sees the same data. Higher overfitting risk.
- `0.8`: Mild randomness. Good balance between diversity and data usage.
- `0.5`: Too aggressive. Each tree only sees half the data, might miss patterns.

---

### 1.8 `colsample_bytree=0.8` — Feature Sampling

**What it does:** Each tree is trained on a random 80% of the features (columns).

```
All features:  [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, Amount]  (11 features)

Tree 1 uses:   [V1, V2, V4, V5, V7, V8, V9, Amount]               (80% = ~9 features)
Tree 2 uses:   [V1, V3, V4, V6, V7, V8, V10, Amount]              (different 9)
Tree 3 uses:   [V2, V3, V5, V6, V7, V9, V10, Amount]              (different 9)
```

**How it reduces correlation between trees:**

Without feature sampling, every tree would probably split on the same "best" feature
first (say, V14 is the strongest fraud signal). All 200 trees would look very similar.
That is like asking the same question 200 times — you get the same answer, not a
better one.

With feature sampling, some trees never see V14 and are forced to find alternative
patterns (maybe V17 combined with Amount). This makes the ensemble more robust because
it does not depend entirely on one feature.

**Combined with `subsample=0.8`:**
Each tree sees 80% of rows AND 80% of columns = only 64% of the total data. This
creates 200 diverse "views" of the data, making the ensemble much stronger than
any single tree.

---

### 1.9 `scale_pos_weight=50` — THE Critical Parameter for Imbalanced Data

**What it does:** Multiplies the loss for positive (fraud) samples by 50. This makes
the model pay 50x more attention to fraud cases during training.

**The formula:**

```
scale_pos_weight = n_negative / n_positive

In our dataset:
  Total transactions:  284,807
  Fraud (positive):    492        (0.173%)
  Legit (negative):    284,315    (99.827%)

  Ideal scale_pos_weight = 284,315 / 492 ≈ 578
```

**Why 50 and not 578?**

Using the exact ratio (578) would make the model scream "FRAUD!" at every remotely
suspicious transaction, maximizing recall but destroying precision. Setting it to 50
is a deliberate choice that says: "Fraud is important, but not 578x more important
than a false alarm."

Think of it as the cost ratio:
- `scale_pos_weight=1`: "Missing a fraud costs the same as a false alarm."
  Result: model predicts almost nothing as fraud (very few positives).
- `scale_pos_weight=50`: "Missing a fraud costs 50x more than a false alarm."
  Result: model is aggressive about catching fraud but still controlled.
- `scale_pos_weight=578`: "Missing a fraud costs 578x more than a false alarm."
  Result: model flags everything remotely suspicious. Many false alarms.

**What happens internally:**

During training, the loss function is:
```
Loss = -1/N * SUM[
    w_i * (y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
]

where:
  w_i = scale_pos_weight  if y_i = 1 (fraud)
  w_i = 1                 if y_i = 0 (legit)
```

A missed fraud (y=1, p=0.01) contributes 50x more to the loss than a false alarm
(y=0, p=0.99). So the gradient pushes the model much harder to catch fraud than
to avoid false alarms.

**Real-world calibration:** In a production system, you would tune `scale_pos_weight`
based on business costs:
```
Cost of missing one fraud:     $5,000 (chargeback + investigation + customer loss)
Cost of one false alarm:       $2    (automated review + customer inconvenience)

Business-optimal weight:       $5,000 / $2 = 2,500
But at that level, precision drops too low for the ops team.
Practical compromise:          50 (catches most fraud, manageable false alarm rate)
```

---

### 1.10 `eval_metric=aucpr` — Why PR-AUC Over ROC-AUC for Imbalanced Data

**What it does:** Tells XGBoost to use Area Under the Precision-Recall Curve as the
evaluation metric during training (for eval_set monitoring, early stopping, etc.).

**Why not ROC-AUC?**

ROC-AUC measures the tradeoff between True Positive Rate and False Positive Rate.
The problem: with 284,315 legitimate transactions, even a 1% false positive rate
means 2,843 false alarms. But ROC-AUC treats this as "99% specificity" — sounds great!

```
ROC-AUC perspective:                   PR-AUC perspective:
  TPR = 400/492 = 81.3%                 Recall = 400/492 = 81.3%
  FPR = 2843/284315 = 1.0%              Precision = 400/(400+2843) = 12.3%
  ↑ Looks great!                         ↑ Terrible! Only 12% of flagged
    "We only bother 1% of legit             transactions are actual fraud.
     customers"                             88% are false alarms.
```

PR-AUC does not give "credit" for correctly classifying the overwhelming majority
of negatives. It focuses entirely on: "Of the things you flagged, how many were
right?" and "Of the real frauds, how many did you catch?"

For imbalanced datasets, PR-AUC is much more informative because it exposes the
cost of false positives in a way that ROC-AUC hides.

---

### 1.11 `n_jobs=-1` — Parallel Training

**What it does:** Use all available CPU cores for training.

```
n_jobs=1:   [Core 1: ████████████████]  (sequential, slow)
            [Core 2: idle.............]
            [Core 3: idle.............]
            [Core 4: idle.............]

n_jobs=-1:  [Core 1: ████]  (parallel, 4x faster)
            [Core 2: ████]
            [Core 3: ████]
            [Core 4: ████]
```

`-1` is a scikit-learn/XGBoost convention meaning "auto-detect and use all cores."
On a machine with 8 cores, this is equivalent to `n_jobs=8`. XGBoost parallelizes
the tree-building process — specifically, it can evaluate multiple split candidates
across features in parallel.

**Note:** The trees themselves are still built sequentially (tree 2 depends on tree 1's
errors). The parallelism is *within* each tree's construction: evaluating "should I
split on V1 > 0.5 or V3 > -1.2?" across multiple features simultaneously.

---

### 1.12 `random_state=42` — Reproducibility Seed

**What it does:** Seeds the random number generator so that `subsample`, `colsample_bytree`,
and any other stochastic operations produce the same result every run.

**Why 42?** It is the "Answer to the Ultimate Question of Life, the Universe, and
Everything" from *The Hitchhiker's Guide to the Galaxy*. It is the most common seed
in data science code. Any integer works; 42 is just convention.

**Why reproducibility matters in MLOps:** If you cannot reproduce a training result,
you cannot debug it, audit it, or trust it. Every random choice (data splitting,
row sampling, feature sampling) must be seeded.

---

### 1.13 Real Example: Hyperparameter Tuning Journey

Here is what a realistic tuning journey might look like:

```
Attempt 1: Default XGBoost (no scale_pos_weight)
  Precision: 0.92    Recall: 0.03   F1: 0.06   AUC-ROC: 0.91
  Diagnosis: Model ignores fraud entirely. Predicts "legit" for everything.
             High precision only because it rarely predicts fraud, and when
             it does, it is very sure.

Attempt 2: Added scale_pos_weight=578 (exact class ratio)
  Precision: 0.05    Recall: 0.98   F1: 0.10   AUC-ROC: 0.93
  Diagnosis: Now flags everything as fraud. Catches almost all fraud but
             also flags thousands of legit transactions.

Attempt 3: scale_pos_weight=50, rest default
  Precision: 0.72    Recall: 0.68   F1: 0.70   AUC-ROC: 0.95
  Diagnosis: Much better! But recall is still low for a fraud system.

Attempt 4: Added subsample=0.8, colsample_bytree=0.8
  Precision: 0.76    Recall: 0.74   F1: 0.75   AUC-ROC: 0.97
  Diagnosis: Sampling reduced overfitting. Metrics improved across the board.

Attempt 5: Tuned max_depth=6, min_child_weight=3, n_estimators=200
  Precision: 0.83    Recall: 0.81   F1: 0.82   AUC-ROC: 0.98
  Diagnosis: This is our final model. Good balance of precision and recall.
```

---

## 2. MLflow Integration — Every Call Explained

### 2.1 Overview of MLflow Calls in train.py

Lines 49-51 set up tracking. Lines 52-118 log everything inside a run context.
Let us go through each call.

### 2.2 `mlflow.set_tracking_uri` (Line 49)

```python
mlflow.set_tracking_uri(str(root / "mlruns"))
```

**What it does:** Tells MLflow where to store experiment data. In our case, a local
directory called `mlruns/` at the project root.

**Where experiments can be stored:**

| URI                              | Storage Type       | Use Case                  |
|----------------------------------|--------------------|---------------------------|
| `./mlruns`                       | Local filesystem   | Development, solo work    |
| `sqlite:///mlflow.db`            | SQLite database    | Small team, single server |
| `http://mlflow-server:5000`      | Remote HTTP server | Team collaboration        |
| `databricks`                     | Databricks hosted  | Enterprise MLflow         |
| `s3://bucket/mlruns`             | AWS S3             | Cloud-native storage      |

**Our choice (local filesystem):** Good for development and CI/CD pipelines. Each DVC
pipeline run stores its own MLflow tracking data locally. For a production team, you
would switch to a remote tracking server so all team members can see each other's
experiments.

**What gets created on disk:**

```
mlruns/
├── 0/                          ← Default experiment
├── 123456789/                  ← Our "fraud-detection" experiment (numeric ID)
│   ├── meta.yaml               ← Experiment metadata
│   └── abc-def-ghi/            ← One run (UUID)
│       ├── meta.yaml           ← Run metadata (start/end time, status)
│       ├── params/             ← One file per parameter
│       │   ├── n_estimators
│       │   ├── max_depth
│       │   └── ...
│       ├── metrics/            ← One file per metric
│       │   ├── val_precision
│       │   ├── val_recall
│       │   └── ...
│       ├── artifacts/          ← Model files, plots, etc.
│       │   └── model/
│       │       ├── model.xgb
│       │       ├── MLmodel
│       │       └── conda.yaml
│       └── tags/               ← Run tags
│           ├── mlflow.runName
│           └── mlflow.source.name
```

### 2.3 `mlflow.set_experiment` (Line 50)

```python
mlflow.set_experiment(params["training"]["experiment_name"])
# Resolves to: mlflow.set_experiment("fraud-detection")
```

**What it does:** Creates or selects an experiment namespace. All subsequent runs will
be grouped under this experiment.

**Why namespace experiments?**

In a real project, you might have multiple experiments:
```
Experiments:
  ├── fraud-detection          ← Production model experiments
  ├── fraud-detection-v2       ← New feature engineering experiments
  ├── fraud-autoencoder        ← Trying a different approach
  └── fraud-detection-tuning   ← Hyperparameter search runs
```

Each experiment keeps its runs organized and comparable. You would not want to compare
an autoencoder's reconstruction error with a classifier's AUC-ROC.

### 2.4 `mlflow.start_run` (Line 52)

```python
with mlflow.start_run(run_name="xgboost-fraud-detection") as run:
```

**What is a "run"?** A single execution of your training code. Each run gets:
- A unique `run_id` (UUID like `a1b2c3d4e5f6`)
- A human-readable `run_name` ("xgboost-fraud-detection")
- Start/end timestamps
- Status (RUNNING, FINISHED, FAILED)

**The `with` context manager pattern:**

```python
with mlflow.start_run(...) as run:
    # Inside here: run is RUNNING
    # All mlflow.log_* calls go to this run
    # If an exception occurs, run status = FAILED
# After exiting: run status = FINISHED (or FAILED)
```

The context manager ensures the run is properly closed even if an error occurs.
Without it, you would need try/finally:

```python
# Equivalent but worse:
run = mlflow.start_run(run_name="xgboost-fraud-detection")
try:
    # ... training code ...
finally:
    mlflow.end_run()  # Must be called or run stays "RUNNING" forever
```

**Run lifecycle:**

```
mlflow.start_run()
     │
     ▼
  RUNNING ──── mlflow.log_params(...)
     │         mlflow.log_metric(...)
     │         mlflow.log_model(...)
     │
     ├─── Success ──► FINISHED
     │
     └─── Exception ──► FAILED
```

### 2.5 `mlflow.log_params` (Line 53)

```python
mlflow.log_params(model_params)
```

**What it does:** Logs all key-value pairs from the `model_params` dictionary as
parameters of the current run. This is a batch call — it logs all parameters at once.

The dictionary looks like:
```python
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 50,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}
```

**Why log hyperparameters?**

1. **Reproducibility:** "What exact settings produced this model?" One click in MLflow UI.
2. **Comparison:** Sort experiments by `max_depth` to see its effect on metrics.
3. **Auditing:** Regulators may require proof of what settings were used.
4. **Debugging:** "Why did last night's model perform badly?" Check its parameters.

**Parameters vs. Metrics:**
- Parameters are **inputs** (set before training). They do not change during training.
- Metrics are **outputs** (measured after training). They are the results.

### 2.6 `mlflow.log_param` — Single Parameter Logging (Lines 54-57)

```python
mlflow.log_param("train_samples", len(X_train))
mlflow.log_param("val_samples", len(X_val))
mlflow.log_param("train_fraud_ratio", float(y_train.mean()))
mlflow.log_param("n_features", X_train.shape[1])
```

**What it does:** Logs individual parameters that are not part of the model config
but are important for understanding the training context.

These are **data parameters** — they describe the dataset, not the model:

| Parameter          | Example Value | Why Log It                                      |
|--------------------|---------------|-------------------------------------------------|
| `train_samples`    | 199,364       | Was the model trained on enough data?            |
| `val_samples`      | 28,481        | Is the validation set large enough?              |
| `train_fraud_ratio`| 0.00173       | Did the class balance change between runs?       |
| `n_features`       | 29            | Were features added or removed?                  |

**Interview insight:** Logging data parameters catches a common production bug: data
pipeline changes that silently alter the training data. If suddenly `train_samples`
drops by 50%, you know something broke upstream before you even look at metrics.

### 2.7 `mlflow.log_metric` with `step` — Time Series Metrics (Line 88)

```python
mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)
```

**What it does:** Logs a metric value at a specific step. This creates a time series
that you can plot in MLflow UI.

```
MLflow UI plot for cv_auc_roc:

AUC-ROC
1.00 ┤
0.98 ┤          ●
0.97 ┤     ●         ●    ●
0.96 ┤ ●
0.95 ┤
     └──┬──┬──┬──┬──┬──
        0  1  2  3  4     fold (step)
```

Without the `step` parameter, calling `log_metric` multiple times with the same
metric name would just overwrite the previous value. With `step`, each call adds
a new data point to the metric's history.

**Use cases for step-based logging:**
- Cross-validation folds (our use): metrics per fold
- Training epochs: loss per epoch
- Batch-level metrics: accuracy per mini-batch
- Time-based monitoring: model performance per day/week

### 2.8 `mlflow.xgboost.log_model` (Line 98)

```python
mlflow.xgboost.log_model(model, "model", input_example=X_val.head(1))
```

**What it does:** Serializes the trained XGBoost model and saves it as an MLflow artifact.

**The three arguments:**
1. `model` — the trained XGBClassifier object
2. `"model"` — the artifact directory name (stored under `artifacts/model/`)
3. `input_example=X_val.head(1)` — a sample input row

**What `input_example` does:**

It saves one row of input data alongside the model. This serves three purposes:

1. **Schema inference:** MLflow generates a model signature (input types, output types)
   so consumers know the expected input format.
2. **Testing:** A quick sanity check — "can this model actually make a prediction?"
3. **Documentation:** Anyone loading the model can see what a valid input looks like.

The resulting artifact directory:
```
artifacts/model/
├── model.xgb              ← Serialized XGBoost model (binary format)
├── MLmodel                ← MLflow model metadata (flavors, signature)
├── conda.yaml             ← Conda environment for reproducibility
├── python_env.yaml        ← Python virtualenv specification
├── requirements.txt       ← pip requirements
└── input_example.json     ← The sample input row from X_val.head(1)
```

**Why `mlflow.xgboost.log_model` instead of `mlflow.sklearn.log_model`?**

MLflow has "flavors" — format-specific serializers. The XGBoost flavor uses the native
`.xgb` format which is faster and smaller than pickle. It also logs XGBoost-specific
metadata. However, since XGBClassifier also implements the scikit-learn API,
`mlflow.sklearn.log_model` would also work (using pickle).

### 2.9 `run.info.run_id` — Unique Run Identification (Line 111)

```python
"run_id": run.info.run_id,
```

**What it does:** Gets the unique identifier for this MLflow run.

The run_id is a UUID string like `"a1b2c3d4e5f67890a1b2c3d4e5f67890"`. It is used to:
1. **Reference the run later:** `mlflow.get_run(run_id)` to load its data
2. **Load the model:** `mlflow.xgboost.load_model(f"runs:/{run_id}/model")`
3. **Link artifacts:** The train_metrics.json file includes the run_id so you can
   trace back from the saved metrics to the full MLflow run with all details.
4. **CI/CD integration:** Pipeline logs can include the run_id for debugging.

### 2.10 Real Example: Comparing 5 Experiments in MLflow UI

Imagine you ran 5 experiments with different configurations:

```
MLflow Experiments Table:
┌─────────────┬──────────┬────────┬────────────────┬──────────┬───────┬─────────┐
│ Run Name    │max_depth │ lr     │scale_pos_weight│ val_prec │ val_f1│ val_auc │
├─────────────┼──────────┼────────┼────────────────┼──────────┼───────┼─────────┤
│ baseline    │ 3        │ 0.3    │ 1              │ 0.95     │ 0.05  │ 0.91    │
│ weighted    │ 3        │ 0.3    │ 578            │ 0.04     │ 0.08  │ 0.92    │
│ balanced    │ 6        │ 0.1    │ 50             │ 0.76     │ 0.75  │ 0.97    │
│ deep-trees  │ 12       │ 0.1    │ 50             │ 0.78     │ 0.74  │ 0.96    │
│ final ★     │ 6        │ 0.1    │ 50             │ 0.83     │ 0.82  │ 0.98    │
└─────────────┴──────────┴────────┴────────────────┴──────────┴───────┴─────────┘
```

From this table you can see:
- "baseline" has excellent precision but terrible recall/F1 (ignores fraud)
- "weighted" overcorrected, now flags too much (terrible precision)
- "balanced" found a sweet spot
- "deep-trees" overfits slightly (worse than balanced despite more depth)
- "final" added subsample/colsample and tuned min_child_weight (best overall)

In the MLflow UI, you would select these 5 runs, click "Compare," and see parallel
coordinates plots, metric comparison charts, and parameter tables — all automatically.

---

## 3. Cross-Validation — Explained in Detail

### 3.1 What is K-Fold Cross-Validation?

K-fold CV splits the training data into K equal parts and trains K separate models,
each time using K-1 parts for training and 1 part for validation.

**5-Fold Split Diagram:**

```
Full Training Data: [████████████████████████████████████████████████████]

Fold 1: [VALIDATE ██████████] [TRAIN ██████████████████████████████████████]
Fold 2: [TRAIN ██████████████] [VALIDATE ██████████] [TRAIN ████████████████]
Fold 3: [TRAIN ████████████████████████████] [VALIDATE ██████████] [TRAIN ██]
Fold 4: [TRAIN ██████████████████████████████████████] [VALIDATE ██████████ ]
Fold 5: [TRAIN ██████████████████████████████████████████████████] [VAL ████]

Result: 5 AUC-ROC scores → mean and std
```

**Why not just use a single train/validation split?**

A single split is noisy. Maybe by luck, all the easy-to-detect fraud ended up in
the validation set. Or maybe all the hard cases did. You would get a misleadingly
high or low score.

With 5 folds, every data point gets to be in the validation set exactly once. The
mean score is a much more reliable estimate of true model performance.

### 3.2 Why `StratifiedKFold` is Critical for Imbalanced Data

```python
cv = StratifiedKFold(
    n_splits=params["training"]["cv_folds"],  # 5
    shuffle=True,
    random_state=params["data"]["random_state"],  # 42
)
```

**Regular KFold vs StratifiedKFold:**

```
Dataset: 1000 samples, 5 fraud (0.5%)

Regular KFold (random split):
  Fold 1: 0 fraud out of 200  ← NO FRAUD AT ALL! Cannot compute AUC.
  Fold 2: 3 fraud out of 200  ← Too many (60% of all fraud)
  Fold 3: 1 fraud out of 200
  Fold 4: 1 fraud out of 200
  Fold 5: 0 fraud out of 200  ← Again, no fraud!

StratifiedKFold (preserves class ratio):
  Fold 1: 1 fraud out of 200  (0.5% — same ratio as full dataset)
  Fold 2: 1 fraud out of 200  (0.5%)
  Fold 3: 1 fraud out of 200  (0.5%)
  Fold 4: 1 fraud out of 200  (0.5%)
  Fold 5: 1 fraud out of 200  (0.5%)
```

Stratified sampling ensures each fold has approximately the same percentage of fraud
cases as the full dataset. Without this, some folds might have zero positive examples,
making metrics undefined (division by zero in precision/recall) or meaningless.

**The three parameters:**

- `n_splits=5`: Number of folds. 5 is the standard choice (good bias-variance tradeoff
  for the CV estimate itself). 10 is also common for smaller datasets.
- `shuffle=True`: Randomly shuffle data before splitting. Essential because the original
  data might be ordered by time, and you do not want the first 20% to be all from
  January and the last 20% from December.
- `random_state=42`: Makes the shuffle reproducible. Same seed = same folds every time.

### 3.3 The CV Loop — Line by Line

```python
cv_scores = []                                                          # Line 81
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):# Line 82
    fold_model = xgb.XGBClassifier(**model_params)                      # Line 83
    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],    # Line 84
                   verbose=False)
    fold_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]   # Line 85
    fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_prob)          # Line 86
    cv_scores.append(fold_auc)                                          # Line 87
    mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)                # Line 88
```

**Line 81:** `cv_scores = []` — Initialize an empty list to collect AUC-ROC scores
from each fold.

**Line 82:** `for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):`

This is doing several things:
- `cv.split(X_train, y_train)` yields 5 pairs of (train_indices, validation_indices).
  The `y_train` argument is needed for stratification — the splitter needs to see the
  labels to ensure equal class distribution.
- `enumerate(...)` adds a fold counter (0, 1, 2, 3, 4).
- `(train_idx, val_idx)` unpacks each pair into arrays of row indices.

For example, fold 0 might produce:
```python
train_idx = [0, 1, 2, 4, 5, 6, 7, 9, ...]    # ~159,491 indices (80%)
val_idx   = [3, 8, 14, 23, ...]                # ~39,873 indices (20%)
```

**Line 83:** `fold_model = xgb.XGBClassifier(**model_params)` — Create a FRESH model
for each fold. This is critical: you must not reuse the model from the previous fold
or from the main training. Each fold needs an independent model to get an unbiased
estimate.

**Line 84:** `fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)`
— Train the fresh model on 80% of the data (the training indices for this fold).
`verbose=False` suppresses XGBoost's per-round output.

**Line 85:** `fold_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]`
— Get predicted fraud probabilities for the held-out 20%.
The `[:, 1]` selects column 1 (fraud probability). `predict_proba` returns:
```python
# Shape: (n_samples, 2)
[[0.98, 0.02],   # 98% legit, 2% fraud
 [0.15, 0.85],   # 15% legit, 85% fraud
 [0.99, 0.01],   # 99% legit, 1% fraud
 ...]
```
Column 0 = P(legit), Column 1 = P(fraud). We want column 1.

**Line 86:** `fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_prob)` — Compute
AUC-ROC for this fold's held-out data.

**Line 87:** `cv_scores.append(fold_auc)` — Add the score to our collection.

**Line 88:** `mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)` — Log the score
to MLflow with the fold number as the step (creates a time series).

### 3.4 Mean and Standard Deviation of CV Scores

```python
mlflow.log_metric("cv_auc_roc_mean", float(np.mean(cv_scores)))   # Line 90
mlflow.log_metric("cv_auc_roc_std", float(np.std(cv_scores)))     # Line 91
logger.info(f"CV AUC-ROC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
```

**What `np.mean(cv_scores)` tells you:** The average model performance across all folds.
This is the best single estimate of how the model will perform on unseen data.

**What `np.std(cv_scores)` tells you:** How much the performance varies across folds.

```
GOOD: CV AUC-ROC = 0.9750 +/- 0.0030
  The model performs consistently across all folds.
  Interpretation: "We're confident the true AUC-ROC is between 0.972 and 0.978."

BAD: CV AUC-ROC = 0.9750 +/- 0.0500
  Huge variance! Performance swings wildly between folds.
  Interpretation: "The AUC-ROC could be anywhere from 0.925 to 1.025."
  This means the model is unstable — its performance depends heavily on
  which specific data points are in the training vs validation set.
```

**High std causes:**
- Too few positive examples (each fold's few fraud cases are very different)
- Model overfitting to specific data patterns
- Data leakage in some folds but not others
- Non-stationary data (time-dependent patterns)

### 3.5 Real Example: Great CV but Terrible Test Performance

```
Scenario: Data scientist sees this and celebrates:

  CV AUC-ROC: 0.9950 +/- 0.0010    ← Nearly perfect!

Then runs evaluation on the holdout test set:

  Test AUC-ROC: 0.8200              ← Much worse!

What happened?

Investigation reveals: the feature engineering step used the ENTIRE dataset
(including test set) to compute statistics like mean and standard deviation
for scaling. This is called DATA LEAKAGE.

The CV folds were all "contaminated" with information from the test set,
so they all showed inflated performance. The true test set, which was not
part of the leakage, showed the real (lower) performance.

Fix: ensure ALL feature engineering is done ONLY on the training set,
then applied to validation and test sets. This is exactly what our
pipeline does — the scaler is fit on X_train and transform is applied
to X_val and X_test.
```

**Another example — temporal leakage:**

```
Scenario: Credit card transactions from January to December.
  Random CV split: each fold mixes transactions from all months.
  CV AUC-ROC: 0.9800 +/- 0.0020

  Temporal test (trained on Jan-Oct, tested on Nov-Dec):
  Test AUC-ROC: 0.8900

  Why? Fraud patterns change over time. Random CV "cheats" by using
  December transactions to predict October transactions. In production,
  you always predict the future from the past.

  Fix: Use TimeSeriesSplit instead of StratifiedKFold when data is temporal.
```

---

## 4. Classification Metrics — Every Metric Deep Dive

### 4.1 The Confusion Matrix

From `evaluate.py`, lines 57-65:

```python
cm = confusion_matrix(y_test, y_pred)
cm_data = {
    "true_negative": int(cm[0][0]),
    "false_positive": int(cm[0][1]),
    "false_negative": int(cm[1][0]),
    "true_positive": int(cm[1][1]),
}
```

**The 2x2 confusion matrix for fraud detection:**

```
                        Predicted
                    Legit     Fraud
                ┌──────────┬──────────┐
Actual  Legit   │    TN    │    FP    │
                │  56,850  │     12   │
                ├──────────┼──────────┤
Actual  Fraud   │    FN    │    TP    │
                │    15    │     85   │
                └──────────┴──────────┘
```

**Each quadrant in fraud terms:**

| Cell | Name            | Count  | Meaning                                              |
|------|-----------------|--------|------------------------------------------------------|
| TN   | True Negative   | 56,850 | Legit transaction correctly approved                  |
| FP   | False Positive  | 12     | Legit transaction wrongly flagged as fraud             |
| FN   | False Negative  | 15     | Fraud transaction we MISSED (the dangerous one!)       |
| TP   | True Positive   | 85     | Fraud transaction we correctly caught                  |

**Why the confusion matrix is the foundation:**
Every classification metric is derived from these four numbers. Understanding the
matrix means understanding every metric.

### 4.2 `precision_score` — Of What We Flagged, How Much Was Actually Fraud?

From `evaluate.py`, line 38 (and `train.py`, line 29):

```python
"precision": float(precision_score(y_test, y_pred, zero_division=0)),
```

**Formula:**
```
                    TP                   85
Precision = ───────────────── = ──────────────── = 0.876
                TP + FP              85 + 12
```

**In plain English:** "Of all the transactions our model flagged as fraud, 87.6% were
actually fraudulent."

**When precision matters most:** When the cost of a false alarm is high.

Examples:
- **Email spam filter:** Low precision means legitimate emails go to spam. The user
  misses an important message from their boss. Bad!
- **Fraud investigation team:** Each flagged transaction requires a human investigator
  to review it (costs $50/review). With precision = 0.10, you waste $450 investigating
  9 legitimate transactions for every real fraud caught.
- **Cancer screening:** Low precision means healthy people are told they might have
  cancer, causing unnecessary anxiety, biopsies, and medical costs.

**The `zero_division=0` parameter:** If there are no positive predictions at all
(TP + FP = 0), precision is undefined (0/0). This parameter says: "If that happens,
return 0 instead of raising an error." This is important for imbalanced datasets where
the model might predict zero positives.

### 4.3 `recall_score` — Of All Actual Fraud, How Much Did We Catch?

```python
"recall": float(recall_score(y_test, y_pred, zero_division=0)),
```

**Formula:**
```
                  TP                   85
Recall = ───────────────── = ──────────────── = 0.850
              TP + FN              85 + 15
```

**In plain English:** "Of all 100 actual fraud transactions in the test set, we caught 85."

**When recall matters most:** When the cost of missing a positive case is high.

Examples:
- **Fraud detection:** Missing a fraud means the bank pays the chargeback ($5,000+).
  A bank might tolerate some false alarms but cannot tolerate missing fraud.
- **Medical diagnosis (cancer screening):** Missing a cancer case means delayed
  treatment. The cost is a human life. Recall must be near 100%.
- **Security threat detection:** Missing a real threat could be catastrophic. False
  alarms are annoying but survivable.

**The asymmetry in fraud detection:** Most fraud systems prioritize recall over precision
because:
```
Cost of FP (false alarm):     $2   (automated review, maybe a phone call)
Cost of FN (missed fraud):    $5,000+ (chargeback, investigation, customer loss, reputational damage)

It is 2,500x more expensive to miss a fraud than to flag a legit transaction.
```

### 4.4 `f1_score` — The Harmonic Mean

```python
"f1": float(f1_score(y_test, y_pred, zero_division=0)),
```

**Formula:**
```
         2 * Precision * Recall       2 * 0.876 * 0.850
F1 = ────────────────────────── = ────────────────────── = 0.863
         Precision + Recall          0.876 + 0.850
```

**Why harmonic mean instead of arithmetic mean?**

The arithmetic mean gives equal weight:
```
Arithmetic mean = (0.876 + 0.850) / 2 = 0.863
```

Wait, in this case they are close so the difference is small. Let us see a case where
it matters:

```
Model A: Precision = 1.00, Recall = 0.01 (flags almost nothing, but when it does, it's right)
  Arithmetic mean = (1.00 + 0.01) / 2 = 0.505  ← Looks OK!
  Harmonic mean   = 2 * 1.00 * 0.01 / (1.00 + 0.01) = 0.0198  ← Terrible!

Model B: Precision = 0.50, Recall = 0.50
  Arithmetic mean = (0.50 + 0.50) / 2 = 0.500  ← Same as Model A!
  Harmonic mean   = 2 * 0.50 * 0.50 / (0.50 + 0.50) = 0.500  ← 25x better than A!
```

The harmonic mean **punishes extreme imbalances**. A model that sacrifices one metric
completely (like Model A with recall = 0.01) gets a near-zero F1 score, even though
its arithmetic mean looks decent. This makes F1 a much better single metric for
evaluating classifiers than the average of precision and recall.

**Mathematical intuition:** The harmonic mean is always less than or equal to the
arithmetic mean, and it equals the arithmetic mean only when both values are identical.
The more different the two values are, the more the harmonic mean penalizes you.

### 4.5 `roc_auc_score` — Area Under the ROC Curve

```python
"auc_roc": float(roc_auc_score(y_test, y_prob)),
```

**Note:** This uses `y_prob` (probability scores), NOT `y_pred` (hard 0/1 predictions).

**What is the ROC curve?**

The ROC (Receiver Operating Characteristic) curve plots True Positive Rate (recall)
vs False Positive Rate at every possible classification threshold.

```
Threshold = 0.5 (default):
  If P(fraud) > 0.5 → predict fraud
  TPR = 0.85, FPR = 0.0002

Threshold = 0.3 (more aggressive):
  If P(fraud) > 0.3 → predict fraud
  TPR = 0.93, FPR = 0.001

Threshold = 0.7 (more conservative):
  If P(fraud) > 0.7 → predict fraud
  TPR = 0.70, FPR = 0.00005
```

**The ROC curve diagram:**

```
TPR (Recall)
1.0 ┤                              ●────────── Perfect model
    │                         ●
0.9 ┤                    ●
    │               ●              ← Our model's curve
0.8 ┤          ●
    │     ●
0.7 ┤  ●
    │ ●
0.6 ┤●                              ╱ Random model (diagonal)
    │                              ╱
0.5 ┤                            ╱
    │                          ╱
0.4 ┤                        ╱
    │                      ╱
0.3 ┤                    ╱
    │                  ╱
0.2 ┤                ╱
    │              ╱
0.1 ┤            ╱
    │          ╱
0.0 ┤────────╱──────────────────────
    0.0    0.2    0.4    0.6    0.8    1.0
                    FPR
```

**What AUC represents:**

AUC = the probability that a randomly chosen positive example (fraud) gets a
higher predicted probability than a randomly chosen negative example (legit).

```
AUC = 0.98 means:
  Pick a random fraud transaction and a random legit transaction.
  98% of the time, the model assigns a HIGHER fraud probability
  to the fraud transaction than to the legit one.
```

**AUC interpretation:**
| AUC   | Meaning                              |
|-------|--------------------------------------|
| 0.50  | Random guessing (coin flip)          |
| 0.70  | Poor model                           |
| 0.80  | Fair model                           |
| 0.90  | Good model                           |
| 0.95  | Very good model                      |
| 0.99  | Excellent (or check for data leakage)|
| 1.00  | Perfect (definitely data leakage)    |

### 4.6 `average_precision_score` — PR Curve for Imbalanced Data

```python
"avg_precision": float(average_precision_score(y_test, y_prob)),
```

**What it does:** Computes the area under the Precision-Recall curve. This is the metric
that truly matters for imbalanced classification.

**The Precision-Recall curve:**

```
Precision
1.0 ┤●
    │  ●
0.9 ┤    ●
    │      ●
0.8 ┤        ●                     ← Our model's curve
    │          ●
0.7 ┤            ●
    │              ●
0.6 ┤                ●
    │                  ●
0.5 ┤                    ●
    │
0.4 ┤
    │                                ← Random baseline = 0.00173
0.0 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    (the fraud rate)
    0.0    0.2    0.4    0.6    0.8    1.0
                    Recall
```

**Why better than ROC-AUC for imbalanced data?**

The random baseline for PR-AUC is the positive class prevalence (0.00173 for our
dataset). A PR-AUC of 0.80 means the model is 460x better than random. This makes
differences between good and great models much more visible.

ROC-AUC compresses everything between 0.90 and 1.00 for a good model on imbalanced
data, making it hard to distinguish "good" from "excellent." PR-AUC spreads the
scores across the full [0, 1] range.

```
Comparing two models:
  Model A: ROC-AUC = 0.970,  PR-AUC = 0.60
  Model B: ROC-AUC = 0.975,  PR-AUC = 0.82

ROC-AUC says: "These models are almost identical" (0.5% difference)
PR-AUC says: "Model B is dramatically better" (37% relative improvement)
```

### 4.7 `classification_report` — The Full Picture

From `evaluate.py`, line 49:

```python
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")
```

**Example output:**

```
              precision    recall  f1-score   support

       Legit       1.00      1.00      1.00     56862
       Fraud       0.88      0.85      0.86       100

    accuracy                           1.00     56962
   macro avg       0.94      0.92      0.93     56962
weighted avg       1.00      1.00      1.00     56962
```

**What each column means:**

| Column       | Meaning                                                     |
|--------------|-------------------------------------------------------------|
| `precision`  | Of predictions for this class, how many were correct?        |
| `recall`     | Of actual examples of this class, how many were caught?      |
| `f1-score`   | Harmonic mean of precision and recall for this class         |
| `support`    | Number of actual examples of this class in the test set      |

**What each row means:**

| Row              | Meaning                                                    |
|------------------|------------------------------------------------------------|
| `Legit`          | Metrics for the negative class (treating legit as positive)|
| `Fraud`          | Metrics for the positive class (what we care about)        |
| `accuracy`       | Overall (TP+TN)/(TP+TN+FP+FN). Misleading for imbalanced! |
| `macro avg`      | Simple average of Legit and Fraud metrics                  |
| `weighted avg`   | Average weighted by support (class size)                   |

**Why accuracy is misleading:**

```
Accuracy = (56850 + 85) / 56962 = 0.9995 (99.95%)

A model that predicts EVERYTHING as legit:
Accuracy = 56862 / 56962 = 0.9982 (99.82%)

The "predict all legit" model has 99.82% accuracy! But it catches zero fraud.
This is why we never use accuracy as a primary metric for imbalanced classification.
```

**`macro avg` vs `weighted avg`:**

- `macro avg`: Treats both classes equally. (Legit_F1 + Fraud_F1) / 2 = (1.00 + 0.86) / 2 = 0.93.
  Good for understanding "how well does the model do on EACH class?"
- `weighted avg`: Weights by class size. Since Legit is 99.8% of the data, weighted avg
  is dominated by Legit performance (near 1.00). This is essentially the same as accuracy
  and equally misleading.

### 4.8 The Precision-Recall Tradeoff

Every classifier that outputs probabilities can be turned into a hard classifier by
choosing a threshold. The threshold determines where you draw the line between
"predict positive" and "predict negative."

**Threshold slider visualization:**

```
Threshold = 0.1 (very aggressive):
  Predict fraud if P(fraud) > 0.1
  ┌────────────────────────────────────────────────┐
  │ Precision: ████░░░░░░  0.12                     │  Very low — many false alarms
  │ Recall:    █████████░  0.98                     │  Very high — catch almost everything
  └────────────────────────────────────────────────┘

Threshold = 0.3:
  ┌────────────────────────────────────────────────┐
  │ Precision: ██████░░░░  0.55                     │  Getting better
  │ Recall:    █████████░  0.93                     │  Still high
  └────────────────────────────────────────────────┘

Threshold = 0.5 (default):
  ┌────────────────────────────────────────────────┐
  │ Precision: █████████░  0.88                     │  Good
  │ Recall:    █████████░  0.85                     │  Good
  └────────────────────────────────────────────────┘

Threshold = 0.7:
  ┌────────────────────────────────────────────────┐
  │ Precision: █████████░  0.95                     │  Very high
  │ Recall:    ███████░░░  0.70                     │  Dropping
  └────────────────────────────────────────────────┘

Threshold = 0.9 (very conservative):
  ┌────────────────────────────────────────────────┐
  │ Precision: ██████████  0.99                     │  Nearly perfect
  │ Recall:    ████░░░░░░  0.40                     │  Missing most fraud!
  └────────────────────────────────────────────────┘
```

**The fundamental tradeoff:** You cannot increase both precision and recall simultaneously
by adjusting the threshold. Lowering the threshold catches more fraud (higher recall)
but also flags more legitimate transactions (lower precision). This is not a model flaw
— it is a mathematical property of any binary classifier.

### 4.9 Real Example: Business Impact of Recall vs Precision Choices

**Scenario:** A bank processes 1,000,000 transactions/day. 500 are fraud (~0.05%).

**Choice A: Optimize for Recall = 0.95 (catch almost all fraud)**
```
True Positives:   475 frauds caught         (saved: 475 * $5,000 = $2,375,000)
False Negatives:  25 frauds missed          (lost:  25  * $5,000 = $125,000)
False Positives:  5,000 legit flagged       (cost:  5,000 * $50  = $250,000)
Precision:        475 / (475 + 5000) = 8.7%

Net savings: $2,375,000 - $125,000 - $250,000 = $2,000,000/day
Customer impact: 5,000 customers per day get their card temporarily blocked.
                 Some will leave the bank. Brand damage accumulates.
```

**Choice B: Optimize for Precision = 0.95 (very few false alarms)**
```
True Positives:   350 frauds caught         (saved: 350 * $5,000 = $1,750,000)
False Negatives:  150 frauds missed         (lost:  150 * $5,000 = $750,000)
False Positives:  18 legit flagged          (cost:  18  * $50    = $900)
Precision:        350 / (350 + 18) = 95.1%

Net savings: $1,750,000 - $750,000 - $900 = $999,100/day
Customer impact: Only 18 false blocks per day. Great customer experience.
                 But 150 frauds go through daily.
```

**The real-world answer:** Most banks use a tiered approach:
- Threshold 0.9+ → Auto-block the transaction (high precision, catches obvious fraud)
- Threshold 0.5-0.9 → Send to human review queue
- Threshold 0.3-0.5 → Send an SMS verification to the customer
- Threshold < 0.3 → Approve automatically

---

## 5. Model Evaluation (evaluate.py) — Line by Line

### 5.1 Function Signature and Setup

```python
def evaluate_model() -> dict:                                           # Line 24
    params = load_params()                                              # Line 25
    root = get_project_root()                                           # Line 26
```

**Line 24:** Returns a dictionary of metrics. The `-> dict` is a type hint for
documentation and IDE support.

**Line 25-26:** Load the same `params.yaml` and project root used by the training
pipeline. This ensures evaluation uses the same configuration (file paths, thresholds).

### 5.2 Loading the Saved Model with `joblib.load`

```python
model = joblib.load(root / "models" / "model.pkl")                     # Line 28
```

**What `joblib.load` does:** Deserializes a Python object from a `.pkl` (pickle) file
back into memory. The model was saved in `train.py` with `joblib.dump(model, model_path)`.

**Why `joblib` instead of `pickle`?**
- `joblib` is optimized for large NumPy arrays (which models contain internally)
- It uses memory mapping for large files, reducing memory usage
- It compresses data better for numerical objects
- For small models, the difference is negligible; for models with large arrays
  (random forests with thousands of trees), joblib can be 10x faster

**What is in the `.pkl` file?**
The entire XGBClassifier object: all 200 trees, the learned weights, the
hyperparameters, the feature names — everything needed to make predictions.

**Security note:** Pickle files can execute arbitrary code when loaded. Never load a
pickle file from an untrusted source. This is why MLflow's model format includes a
`conda.yaml` and uses safer serialization when possible.

### 5.3 Loading Test Data

```python
processed_dir = root / params["data"]["processed_path"]                 # Line 30
X_test = pd.read_csv(processed_dir / "X_test.csv")                     # Line 31
y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()            # Line 32
```

**Line 30:** Resolves to `data/processed/` — the same directory where the preprocessing
pipeline saved the split data.

**Line 31:** Loads the feature matrix. `X_test` is a DataFrame where each row is a
transaction and each column is a feature (V1-V28, Amount after scaling).

**Line 32:** Loads the labels. `.squeeze()` converts a single-column DataFrame into a
Series. Without it, `y_test` would be a DataFrame with shape `(n, 1)` instead of a
Series with shape `(n,)`, which can cause subtle bugs with scikit-learn.

```python
# Without squeeze():
y_test = pd.read_csv("y_test.csv")
type(y_test)  # DataFrame
y_test.shape  # (56962, 1)

# With squeeze():
y_test = pd.read_csv("y_test.csv").squeeze()
type(y_test)  # Series
y_test.shape  # (56962,)
```

### 5.4 `model.predict` vs `model.predict_proba`

```python
y_pred = model.predict(X_test)                                          # Line 34
y_prob = model.predict_proba(X_test)[:, 1]                              # Line 35
```

**Line 34 — Hard predictions:**
```python
y_pred = [0, 0, 0, 1, 0, 0, 1, 0, ...]
# Each value is 0 (legit) or 1 (fraud)
# Uses the default threshold of 0.5
# Internally: if predict_proba >= 0.5 → 1, else → 0
```

**Line 35 — Soft predictions (probabilities):**
```python
model.predict_proba(X_test)
# Returns shape (56962, 2):
# [[0.9998, 0.0002],    ← 99.98% legit, 0.02% fraud
#  [0.9543, 0.0457],    ← 95.43% legit, 4.57% fraud
#  [0.1200, 0.8800],    ← 12% legit, 88% fraud
#  ...]

model.predict_proba(X_test)[:, 1]
# Selects column 1 (fraud probability) for all rows:
# [0.0002, 0.0457, 0.8800, ...]
```

**Why we need both:**
- `y_pred` (hard): needed for precision, recall, F1, confusion matrix
  (these require binary decisions)
- `y_prob` (soft): needed for AUC-ROC, average_precision_score, ROC curve
  (these need probability scores to evaluate at all thresholds)

### 5.5 Computing and Saving Metrics

```python
metrics = {                                                             # Lines 37-46
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "auc_roc": float(roc_auc_score(y_test, y_prob)),
    "avg_precision": float(average_precision_score(y_test, y_prob)),
    "test_samples": len(y_test),
    "fraud_count": int(y_test.sum()),
    "predicted_fraud_count": int(y_pred.sum()),
}
```

Every metric is wrapped in `float()` to ensure JSON serializability. NumPy's `float64`
type is not natively JSON-serializable, but Python's `float` is.

The last three entries are not metrics but **metadata**:
- `test_samples`: how large was the test set? (sanity check)
- `fraud_count`: how many actual frauds? (ensures test set was not corrupted)
- `predicted_fraud_count`: how many did the model flag? (quick overview)

### 5.6 Classification Report Logging

```python
logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")
```

The `target_names` parameter replaces the default class labels (0, 1) with human-readable
names. This makes the log output immediately interpretable without knowing that 0=Legit
and 1=Fraud.

### 5.7 ROC Curve Data Generation

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)                                 # Line 67
roc_data = [                                                            # Lines 68-71
    {"fpr": float(fpr[i]), "tpr": float(tpr[i])}
    for i in range(0, len(fpr), max(1, len(fpr) // 100))
]
```

**Line 67:** `roc_curve` computes the FPR and TPR at every possible threshold.

```python
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# fpr = [0.0, 0.00001, 0.00002, ..., 1.0]       (false positive rates)
# tpr = [0.0, 0.01,    0.02,    ..., 1.0]       (true positive rates / recall)
# thresholds = [0.999, 0.998, 0.997, ..., 0.001] (classification thresholds)
#
# Each triple (fpr[i], tpr[i], thresholds[i]) says:
# "If you set the threshold to thresholds[i], your FPR will be fpr[i]
#  and your TPR will be tpr[i]."
```

The `_` discards the thresholds — we only need fpr and tpr for plotting.

**Lines 68-71: Downsampling — why `range(0, len(fpr), max(1, len(fpr) // 100))`**

`roc_curve` returns one point per unique predicted probability. With 56,962 test
samples, that could be ~56,962 points. Saving all of them to JSON would create a
huge file and slow down any visualization.

The downsampling logic:
```python
len(fpr) // 100   # Divide total points by 100 → step size
                   # If len(fpr) = 50000, step = 500
                   # So we keep every 500th point → ~100 points total

max(1, ...)        # If len(fpr) < 100, step would be 0.
                   # max(1, 0) = 1 → keep every point (no downsampling needed)

range(0, len(fpr), step)  # Start at 0, go to end, skip by step
```

Result: approximately 100 evenly-spaced points along the ROC curve. Enough for a
smooth plot, but small enough for a JSON file (~5KB instead of ~2MB).

### 5.8 Quality Gates — The Gatekeeper Pattern

```python
_check_quality_gates(metrics, params)                                   # Line 75
```

And the gate implementation (lines 80-102):

```python
def _check_quality_gates(metrics: dict, params: dict) -> None:
    thresholds = params["thresholds"]
    gates = {
        "recall": ("min_recall", metrics["recall"]),
        "precision": ("min_precision", metrics["precision"]),
        "f1": ("min_f1", metrics["f1"]),
        "auc_roc": ("min_auc_roc", metrics["auc_roc"]),
    }

    failures = []
    for metric_name, (threshold_key, actual) in gates.items():
        minimum = thresholds[threshold_key]
        status = "PASS" if actual >= minimum else "FAIL"
        logger.info(f"  Quality gate {metric_name}: {actual:.4f} >= {minimum} → {status}")
        if actual < minimum:
            failures.append(f"{metric_name}: {actual:.4f} < {minimum}")

    if failures:
        msg = "Quality gates FAILED: " + "; ".join(failures)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("All quality gates PASSED")
```

**What are quality gates?**

Quality gates are automated checks that prevent a bad model from moving forward in
the pipeline. They are the "bouncer at the door" — your model must pass all gates
or it does not get deployed.

**The thresholds from `params.yaml`:**

```yaml
thresholds:
  min_recall: 0.02
  min_precision: 0.50
  min_f1: 0.03
  min_auc_roc: 0.90
```

**How each gate works:**

```
Gate Check:
  ┌─────────────┬──────────┬──────────┬──────────┐
  │ Metric      │ Actual   │ Minimum  │ Status   │
  ├─────────────┼──────────┼──────────┼──────────┤
  │ recall      │ 0.8500   │ 0.02     │ PASS ✓   │
  │ precision   │ 0.8760   │ 0.50     │ PASS ✓   │
  │ f1          │ 0.8628   │ 0.03     │ PASS ✓   │
  │ auc_roc     │ 0.9800   │ 0.90     │ PASS ✓   │
  └─────────────┴──────────┴──────────┴──────────┘
  Result: All gates PASSED → model proceeds to deployment
```

**The underscore prefix `_check_quality_gates`:** The leading underscore is a Python
convention meaning "this function is private/internal." It is only called by
`evaluate_model()` and is not part of the public API.

**Why these specific thresholds are low:** The thresholds in params.yaml (recall >= 0.02,
F1 >= 0.03) are intentionally set low for the learning project to ensure the pipeline
runs successfully. In production, you would set them much higher:

```yaml
# Production thresholds:
thresholds:
  min_recall: 0.75
  min_precision: 0.50
  min_f1: 0.60
  min_auc_roc: 0.95
```

### 5.9 `raise ValueError` — How This Stops the CI/CD Pipeline

```python
if failures:
    msg = "Quality gates FAILED: " + "; ".join(failures)
    logger.error(msg)
    raise ValueError(msg)
```

**What happens when `raise ValueError` is called:**

1. The Python process exits with a non-zero exit code (exit code 1)
2. DVC sees the non-zero exit code and marks the pipeline stage as FAILED
3. The CI/CD system (GitHub Actions) sees the DVC failure and marks the pipeline as FAILED
4. The deployment step never runs
5. The pull request gets a red X instead of a green checkmark
6. The team is notified that model quality degraded

```
Pipeline Flow:
  [data prep] → [feature eng] → [train] → [evaluate] → [deploy]
                                               │
                                        raise ValueError!
                                               │
                                               ✗ PIPELINE STOPS
                                               │
                                        [deploy] NEVER RUNS
                                               │
                                        Bad model never reaches production
```

**This is the safety net.** Without quality gates, a bad model could silently make it
to production. With quality gates, the pipeline is self-protecting.

### 5.10 The `failures` Collection Pattern

```python
failures = []
for metric_name, (threshold_key, actual) in gates.items():
    minimum = thresholds[threshold_key]
    ...
    if actual < minimum:
        failures.append(f"{metric_name}: {actual:.4f} < {minimum}")

if failures:
    raise ValueError(...)
```

**Why collect all failures instead of raising on the first one?**

If you raise immediately on the first failure, the developer only sees:
```
Quality gates FAILED: recall: 0.0100 < 0.02
```

But the model might also fail precision, F1, and AUC-ROC. By collecting all failures
first, the developer sees the full picture:
```
Quality gates FAILED: recall: 0.0100 < 0.02; precision: 0.3000 < 0.50; f1: 0.0194 < 0.03
```

This saves debugging time — you immediately know the model failed on 3 out of 4
gates, suggesting a fundamental problem (not just a marginal miss on one metric).

### 5.11 Real Example: Quality Gate Catches Corrupted Data

```
Scenario: A nightly batch job updates the training data. Due to a bug in the
data pipeline, the "Amount" feature was set to 0 for all transactions.

Without quality gates:
  1. Model trains on corrupted data
  2. Model gets deployed automatically
  3. In production, it cannot use Amount for predictions
  4. Fraud detection rate drops from 85% to 40%
  5. 3 days later, the fraud team notices increased chargebacks
  6. Total cost: $2.3M in missed fraud over 3 days

With quality gates:
  1. Model trains on corrupted data
  2. Evaluate runs → AUC-ROC = 0.82 (normally 0.98)
  3. Quality gate: 0.82 < 0.90 → FAIL
  4. Pipeline stops. Alert sent to team.
  5. Engineer investigates, finds the Amount bug, fixes it
  6. Total cost: $0 (old model stays in production during investigation)
```

Another example:

```
Scenario: Someone accidentally runs training on a filtered dataset
(only transactions from one merchant).

  Quality gate results:
    recall:    0.9500 >= 0.02    → PASS   (high because simple data)
    precision: 0.0300 >= 0.50    → FAIL   (terrible because model
                                           learned ONE merchant's pattern,
                                           flags everything else as fraud)
    f1:        0.0582 >= 0.03    → PASS
    auc_roc:   0.8700 >= 0.90    → FAIL

  Quality gates FAILED: precision: 0.0300 < 0.50; auc_roc: 0.8700 < 0.90

  The quality gate caught it. The model with 3% precision would have
  created thousands of false alarms per hour in production.
```

---

## 6. Feature Importance — The Bonus Insight

Lines 94-96 of `train.py`:

```python
feature_importance = dict(zip(X_train.columns, model.feature_importances_.tolist()))
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
logger.info(f"Top features: {top_features}")
```

**Line 94:**

- `model.feature_importances_`: A NumPy array with one importance score per feature.
  XGBoost computes importance based on how much each feature improved the model
  (default: "gain" = average improvement in loss when a feature is used in a split).
- `X_train.columns`: The feature names (V1, V2, ..., V28, Amount).
- `zip(...)`: Pairs each name with its importance: `[("V1", 0.05), ("V2", 0.03), ...]`
- `dict(...)`: Converts to a dictionary for easy lookup.
- `.tolist()`: Converts NumPy array to Python list (for JSON serializability).

**Line 95:**

- `sorted(..., key=lambda x: x[1], reverse=True)`: Sort by importance value, highest first.
- `[:10]`: Keep only the top 10 features.

**Why log feature importance?**

1. **Model interpretability:** "Why did the model flag this transaction?" If V14 is the
   top feature, and V14 represents "deviation from average transaction amount," you
   can explain the model's reasoning.
2. **Feature engineering guidance:** Low-importance features can be removed (simplifies
   the model). High-importance features suggest where to invest in better data.
3. **Drift detection:** If feature importance changes drastically between model versions,
   something fundamental changed in the data.

---

## 7. The Full Training Pipeline — Putting It All Together

Here is the complete flow from start to finish:

```
train_model() called
     │
     ├── Load params.yaml configuration
     ├── Load preprocessed CSV files (X_train, y_train, X_val, y_val)
     │
     ├── Set up MLflow tracking
     │   ├── Set tracking URI (local mlruns/ directory)
     │   └── Set experiment name ("fraud-detection")
     │
     ├── Start MLflow run
     │   ├── Log all hyperparameters (n_estimators, max_depth, etc.)
     │   ├── Log data parameters (train_samples, fraud_ratio, etc.)
     │   │
     │   ├── Create and train XGBClassifier
     │   │   └── model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
     │   │
     │   ├── Evaluate on validation set
     │   │   ├── predict → hard labels
     │   │   ├── predict_proba → probabilities
     │   │   ├── Compute 5 metrics (precision, recall, F1, AUC-ROC, AP)
     │   │   └── Log all metrics to MLflow
     │   │
     │   ├── Run 5-fold stratified cross-validation
     │   │   ├── Fold 0: train on 80%, evaluate on 20% → AUC-ROC
     │   │   ├── Fold 1: train on 80%, evaluate on 20% → AUC-ROC
     │   │   ├── Fold 2: train on 80%, evaluate on 20% → AUC-ROC
     │   │   ├── Fold 3: train on 80%, evaluate on 20% → AUC-ROC
     │   │   ├── Fold 4: train on 80%, evaluate on 20% → AUC-ROC
     │   │   └── Log mean and std of CV scores
     │   │
     │   ├── Extract and log feature importance (top 10)
     │   │
     │   ├── Save model artifact to MLflow (with input example)
     │   ├── Save model to models/model.pkl (with joblib)
     │   └── Save metrics to metrics/train_metrics.json (with run_id)
     │
     └── Return model path

evaluate_model() called (next DVC stage)
     │
     ├── Load model from models/model.pkl
     ├── Load test data (X_test, y_test)
     │
     ├── Make predictions (hard and soft)
     ├── Compute 5 metrics + metadata
     ├── Log classification report
     │
     ├── Save eval_metrics.json
     ├── Save confusion_matrix.json
     ├── Save roc_curve.json (downsampled to ~100 points)
     │
     ├── Run quality gates
     │   ├── recall >= 0.02?     → PASS/FAIL
     │   ├── precision >= 0.50?  → PASS/FAIL
     │   ├── f1 >= 0.03?         → PASS/FAIL
     │   └── auc_roc >= 0.90?    → PASS/FAIL
     │
     ├── All pass? → Return metrics (pipeline continues to deploy)
     └── Any fail? → raise ValueError (pipeline stops, no deployment)
```

---

## 8. Interview Questions You Should Be Able to Answer

After reading this guide, you should be able to answer:

1. **"Why did you choose XGBoost over Random Forest or Logistic Regression?"**
   XGBoost handles imbalanced data natively (scale_pos_weight), has built-in
   regularization (max_depth, min_child_weight), and typically outperforms other
   methods on tabular data. It also supports parallel training (n_jobs=-1).

2. **"Your model has 95% accuracy. Is that good?"**
   For imbalanced classification, accuracy is meaningless. A model predicting all
   transactions as legit gets 99.8% accuracy. We use F1, PR-AUC, and recall.

3. **"How would you choose between precision and recall?"**
   It depends on business costs. In fraud detection, missing fraud costs ~$5,000 and
   a false alarm costs ~$2. So recall is prioritized. But in practice, we use a tiered
   threshold approach: auto-block for very high confidence, human review for medium
   confidence.

4. **"What is the purpose of cross-validation if you already have a validation set?"**
   The validation set gives a single point estimate. CV gives a distribution (mean +/- std).
   If CV std is high, the model is unstable. CV also uses the full training data more
   efficiently (every point gets validated).

5. **"How do quality gates fit into your CI/CD pipeline?"**
   Quality gates are automated checks in the evaluate stage. If any metric falls below
   its threshold, `raise ValueError` stops the pipeline with a non-zero exit code.
   The deployment stage never runs, and the team is alerted. The previous model
   stays in production.

6. **"Why do you log the run_id in train_metrics.json?"**
   Traceability. If a deployed model misbehaves, we can trace from the serving endpoint
   back to the exact MLflow run, see all hyperparameters, data parameters, and metrics,
   and understand exactly what happened during training.

7. **"What does scale_pos_weight=50 do and how did you choose 50?"**
   It multiplies the loss for positive samples by 50, making the model 50x more
   sensitive to missing fraud than to false alarms. The theoretical value is
   n_negative/n_positive (~578), but that is too aggressive. 50 was chosen through
   hyperparameter tuning as the best balance of precision and recall.

8. **"How would you move from local MLflow tracking to a team setup?"**
   Change `mlflow.set_tracking_uri` from a local path to a remote server URL
   (e.g., `http://mlflow.company.com:5000`). Set up a shared database backend
   (PostgreSQL) and artifact store (S3). All team members point to the same server.
