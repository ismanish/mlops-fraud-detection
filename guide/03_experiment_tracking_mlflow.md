# Experiment Tracking with MLflow

## Why Experiment Tracking Matters

Machine learning development is inherently experimental. You try different algorithms,
hyperparameters, feature sets, data preprocessing strategies, and training procedures.
Without a system to track these experiments, you end up in one of these situations:

1. **The lost experiment.** "I got 0.97 AUC-ROC two weeks ago, but I can't remember which
   hyperparameters I used." The result is gone forever.

2. **The spreadsheet of doom.** A shared Google Sheet with columns like "max_depth",
   "n_estimators", "AUC", "notes". It works for 20 experiments but becomes unmanageable
   at 200. No one updates it consistently.

3. **The notebook graveyard.** Dozens of Jupyter notebooks named `train_v2_final.ipynb`,
   `train_v2_final_FINAL.ipynb`, `train_v3_johns_version.ipynb`. Which one produced the
   production model? Nobody knows.

Experiment tracking solves all of these by automatically recording:
- **What** you ran (parameters, code version)
- **How** it performed (metrics)
- **What** it produced (model artifacts, plots)
- **When** it ran (timestamps)
- **Who** ran it (user, environment)

---

## MLflow Components

MLflow is an open-source platform with four main components:

```
+------------------------------------------------------------------+
|                          MLflow                                   |
|                                                                   |
|  +----------------+  +----------------+  +---------------------+ |
|  |   Tracking     |  |    Models      |  |     Registry        | |
|  |                |  |                |  |                     | |
|  | - Parameters   |  | - Model format |  | - Model versions   | |
|  | - Metrics      |  | - Flavors      |  | - Stage transitions| |
|  | - Artifacts    |  | - Signatures   |  | - Approvals        | |
|  | - Tags         |  | - Input/output |  | - Annotations      | |
|  | - Source code  |  |   examples     |  |                     | |
|  +----------------+  +----------------+  +---------------------+ |
|                                                                   |
|  +----------------+                                               |
|  |   Projects     |                                               |
|  |                |                                               |
|  | - MLproject    |                                               |
|  |   file         |                                               |
|  | - conda.yaml   |                                               |
|  | - Reproducible |                                               |
|  |   runs         |                                               |
|  +----------------+                                               |
+------------------------------------------------------------------+
```

### 1. MLflow Tracking

The core component. Records experiments as "runs" organized into "experiments." Each run
captures parameters, metrics, artifacts, and tags.

**How our project uses it** (`src/models/train.py`):

```python
mlflow.set_tracking_uri(str(root / "mlruns"))
mlflow.set_experiment(params["training"]["experiment_name"])  # "fraud-detection"

with mlflow.start_run(run_name="xgboost-fraud-detection") as run:
    # Log all hyperparameters from params.yaml
    mlflow.log_params(model_params)  # n_estimators, max_depth, learning_rate, etc.
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("val_samples", len(X_val))
    mlflow.log_param("train_fraud_ratio", float(y_train.mean()))
    mlflow.log_param("n_features", X_train.shape[1])

    # Train the model...

    # Log validation metrics
    for name, value in val_metrics.items():
        mlflow.log_metric(f"val_{name}", value)
    # Logged: val_precision, val_recall, val_f1, val_auc_roc, val_avg_precision

    # Log cross-validation metrics with step tracking
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        # ... compute fold_auc ...
        mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)

    mlflow.log_metric("cv_auc_roc_mean", float(np.mean(cv_scores)))
    mlflow.log_metric("cv_auc_roc_std", float(np.std(cv_scores)))

    # Log the model as an artifact with input example
    mlflow.xgboost.log_model(model, "model", input_example=X_val.head(1))
```

Key points:
- `mlflow.log_params()` accepts a dictionary, logging all key-value pairs at once
- `mlflow.log_metric()` with `step=fold` creates a metric history (useful for plotting
  CV fold performance)
- `mlflow.xgboost.log_model()` uses the XGBoost "flavor" for native model serialization
- `input_example=X_val.head(1)` stores a sample input for documentation and signature
  inference

### 2. MLflow Models

MLflow Models provide a standard format for packaging ML models that can be deployed to
various serving platforms.

**Model flavors** are adapters for different ML frameworks:
- `mlflow.xgboost` --- Our project uses this
- `mlflow.sklearn` --- For scikit-learn models
- `mlflow.pytorch` --- For PyTorch models
- `mlflow.tensorflow` --- For TensorFlow/Keras models
- `mlflow.pyfunc` --- Generic Python function (works with any model)

When we call `mlflow.xgboost.log_model(model, "model")`, MLflow creates:

```
mlruns/<experiment_id>/<run_id>/artifacts/model/
|-- MLmodel              # Metadata: flavors, signature, input example
|-- conda.yaml           # Conda environment for reproducibility
|-- model.xgb            # Serialized XGBoost model (native format)
|-- python_env.yaml      # Python environment specification
|-- requirements.txt     # pip requirements
|-- input_example.json   # Sample input we provided
```

The `MLmodel` file describes how to load and serve the model:
```yaml
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.xgboost
    model_path: model.xgb
    python_version: 3.12.0
  xgboost:
    code: null
    data: model.xgb
    model_class: xgboost.core.Booster
    xgb_version: 2.0.3
```

### 3. MLflow Model Registry

The Model Registry is a centralized model store with versioning, stage transitions, and
annotations. It provides a workflow for promoting models from experimentation to production.

```
Model Registry Workflow:

  Experiment Run           Model Registry            Production
  +-------------+         +------------------+       +------------+
  | Run #47     | ------> | fraud-detector   |       |            |
  | AUC: 0.97   |  register| v1 (Staging)   | ----> | Serving    |
  | Recall: 0.85 |        | v2 (Production) |promote| (Lambda)   |
  +-------------+         | v3 (Archived)   |       |            |
                           +------------------+       +------------+
```

**Stage transitions:**
- **None** --- Just registered, not assigned to any stage
- **Staging** --- Under review, running integration tests
- **Production** --- Actively serving predictions
- **Archived** --- Retired, kept for audit trail

In our project, we do not use the Model Registry directly (we use a simpler deploy
pipeline), but it would be the natural next step for managing multiple model versions in
a team setting:

```python
# Example: registering a model (not currently in our code, but straightforward to add)
import mlflow

# During training
with mlflow.start_run() as run:
    mlflow.xgboost.log_model(model, "model", registered_model_name="fraud-detector")

# Later: promote to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud-detector",
    version=2,
    stage="Production"
)

# At serving time: load the production model
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
```

### 4. MLflow Projects

MLflow Projects package code + environment as a reproducible unit. Defined by an `MLproject`
file:

```yaml
name: fraud-detection
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      max_depth: {type: int, default: 6}
      n_estimators: {type: int, default: 200}
    command: "python -m src.models.train"
```

We do not use MLflow Projects because we use DVC for pipeline orchestration and Docker for
environment reproducibility. MLflow Projects overlaps with both of those tools. The choice
is a matter of preference and team workflow.

---

## How Our Project Uses MLflow: Complete Walkthrough

### Tracking URI and Experiment Setup

```python
# In src/models/train.py
mlflow.set_tracking_uri(str(root / "mlruns"))
mlflow.set_experiment(params["training"]["experiment_name"])  # "fraud-detection"
```

The tracking URI is set to a local directory `mlruns/` in the project root. This means
MLflow stores all experiment data as files on disk. For a team setting, you would point
this to a remote tracking server:

```python
# Team setting (not our current setup, but a natural evolution)
mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")
```

### What Gets Logged Per Run

Every training run in our project logs the following:

```
Parameters (15+):
  n_estimators:      200
  max_depth:         6
  learning_rate:     0.1
  min_child_weight:  3
  subsample:         0.8
  colsample_bytree:  0.8
  scale_pos_weight:  50
  eval_metric:       aucpr
  random_state:      42
  n_jobs:            -1
  train_samples:     199364
  val_samples:       28481
  train_fraud_ratio: 0.0017
  n_features:        29

Metrics (9+):
  val_precision:     0.xxxx
  val_recall:        0.xxxx
  val_f1:            0.xxxx
  val_auc_roc:       0.xxxx
  val_avg_precision: 0.xxxx
  cv_auc_roc:        0.xxxx  (per fold, 5 steps)
  cv_auc_roc_mean:   0.xxxx
  cv_auc_roc_std:    0.xxxx

Artifacts:
  model/             (XGBoost model in MLflow format)
    MLmodel
    model.xgb
    conda.yaml
    requirements.txt
    input_example.json
```

### Viewing the MLflow UI

```bash
# Start the MLflow UI
mlflow ui --port 5000
# Or use our Makefile shortcut
make mlflow-ui
```

The UI is accessible at `http://localhost:5000` and provides:

```
+------------------------------------------------------------------+
| MLflow                                    fraud-detection        |
+------------------------------------------------------------------+
| Experiments  |  Runs                                              |
|              |                                                    |
| > fraud-     |  Run Name              | val_auc_roc | val_recall |
|   detection  |  ----------------------|-------------|------------|
|              |  xgboost-fraud-det... | 0.9721      | 0.8234     |
|              |  xgboost-fraud-det... | 0.9685      | 0.7891     |
|              |  xgboost-fraud-det... | 0.9754      | 0.8456     |
|              |                                                    |
|              |  [Compare] [Delete] [Search]                      |
+------------------------------------------------------------------+

Clicking a run shows:
+------------------------------------------------------------------+
| Run: xgboost-fraud-detection                                     |
| Run ID: abc123def456                                             |
| Duration: 2m 34s                                                 |
+------------------------------------------------------------------+
| Parameters          | Metrics              | Artifacts           |
|---------------------|----------------------|---------------------|
| n_estimators: 200   | val_auc_roc: 0.9721  | model/              |
| max_depth: 6        | val_recall: 0.8234   |   MLmodel           |
| learning_rate: 0.1  | val_f1: 0.7891       |   model.xgb         |
| subsample: 0.8      | cv_auc_roc_mean:     |   conda.yaml        |
| ...                 |   0.9698             |   requirements.txt  |
|                     | cv_auc_roc_std:      |   input_example.json|
|                     |   0.0023             |                     |
+------------------------------------------------------------------+
```

### Comparing Experiments

The MLflow UI lets you select multiple runs and compare them side-by-side:

```
+------------------------------------------------------------------+
| Compare Runs (3 selected)                                        |
+------------------------------------------------------------------+
|                  | Run 1      | Run 2      | Run 3               |
|------------------|------------|------------|---------------------|
| max_depth        | 6          | 8          | 4                   |
| n_estimators     | 200        | 200        | 300                 |
| learning_rate    | 0.1        | 0.05       | 0.1                 |
| val_auc_roc      | 0.9721     | 0.9754     | 0.9698              |
| val_recall       | 0.8234     | 0.8456     | 0.8012              |
| val_f1           | 0.7891     | 0.8023     | 0.7756              |
| cv_auc_roc_mean  | 0.9698     | 0.9731     | 0.9672              |
+------------------------------------------------------------------+
```

This makes it immediately clear which hyperparameter combination performed best.

---

## MLflow Tracking Architecture

```
Local Development:

  train.py -----> mlruns/ (local directory)
                     |
                     +-- experiment_id/
                          +-- run_id_1/
                          |    +-- meta.yaml
                          |    +-- params/
                          |    +-- metrics/
                          |    +-- artifacts/
                          +-- run_id_2/
                               +-- ...

Team / Production Setup:

  train.py -----> MLflow Tracking Server -----> Backend Store (PostgreSQL)
                         |                            |
                         |                     Stores: params, metrics,
                         |                     tags, run metadata
                         |
                         +-----> Artifact Store (S3)
                                      |
                                Stores: models, plots,
                                data samples, custom artifacts
```

Our project uses the local file-based tracking store (`mlruns/` directory). This is
listed in `.gitignore` because MLflow runs should not be committed to Git --- they can
be large and are better managed separately.

For production use, you would deploy an MLflow Tracking Server:

```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@db:5432/mlflow \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000
```

---

## Integration: MLflow + DVC

Our project uses both MLflow and DVC, and they serve complementary purposes:

```
+-------------------+          +-------------------+
|       DVC         |          |      MLflow       |
+-------------------+          +-------------------+
| Data versioning   |          | Experiment params |
| Pipeline DAG      |          | Training metrics  |
| File-level hashes |          | Model artifacts   |
| Smart caching     |          | Run comparison    |
| S3 data remote    |          | UI dashboard      |
+-------------------+          +-------------------+
        |                              |
        |     Overlap: Model Storage   |
        +------------------------------+
        |
        models/model.pkl
        (DVC tracks the file, MLflow logs the artifact)
```

There is intentional overlap: both DVC (via `outs: models/model.pkl`) and MLflow (via
`mlflow.xgboost.log_model()`) track the trained model. This is not redundancy --- it
serves different purposes:

- **DVC** tracks `model.pkl` as part of the pipeline DAG. Changing hyperparameters
  triggers retraining. The model is versioned alongside its training data.
- **MLflow** logs the model as an experiment artifact. It captures the model alongside
  all the metrics and parameters from that specific training run. This enables comparison
  and selection.

---

## MLflow Programmatic Access

Beyond the UI, MLflow provides a Python API for querying and managing experiments:

```python
import mlflow

# Get the tracking client
client = mlflow.tracking.MlflowClient()

# List all experiments
for exp in client.search_experiments():
    print(f"{exp.name}: {exp.experiment_id}")

# Search runs with filters
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_auc_roc > 0.95",
    order_by=["metrics.val_auc_roc DESC"],
    max_results=10,
)

# Get the best run
best_run = runs[0]
print(f"Best AUC-ROC: {best_run.data.metrics['val_auc_roc']}")
print(f"Parameters: {best_run.data.params}")
print(f"Run ID: {best_run.info.run_id}")

# Load the best model
model = mlflow.xgboost.load_model(f"runs:/{best_run.info.run_id}/model")
```

This is powerful for automation: you could write a script that finds the best model
across all experiments and promotes it to production.

---

## Comparison: MLflow vs Alternatives

| Feature              | MLflow           | W&B (Weights & Biases) | Neptune          | CometML          |
|----------------------|------------------|------------------------|------------------|------------------|
| **Open source**      | Yes (Apache 2.0) | No (SaaS)              | No (SaaS)        | Partially        |
| **Self-hosted**      | Yes              | Yes (Enterprise)       | Yes (Enterprise) | Yes (Enterprise) |
| **Free tier**        | Unlimited (self) | 100 GB storage         | Limited          | Limited          |
| **Experiment tracking** | Yes           | Yes                    | Yes              | Yes              |
| **Model registry**   | Yes             | Yes (Artifacts)        | Yes              | Yes              |
| **Hyperparameter tuning** | No (use Optuna) | Sweeps (built-in)  | Built-in         | Built-in         |
| **Team collaboration** | Basic          | Excellent              | Excellent        | Good             |
| **Real-time logging** | No             | Yes (live dashboard)   | Yes              | Yes              |
| **System metrics**   | No              | Yes (GPU, CPU, RAM)    | Yes              | Yes              |
| **Data versioning**  | Limited          | Artifacts              | No               | No               |
| **Pipeline support** | Projects         | Launch                 | No               | No               |
| **Learning curve**   | Low              | Low                    | Medium           | Low              |
| **Integration with DVC** | Excellent   | Good                   | Good             | Good             |

### When to use what

- **MLflow:** Best for teams that want full control, self-hosting, and no vendor lock-in.
  Pairs well with DVC for data versioning. Best for companies with data residency requirements
  or tight budgets. This is what we use.

- **W&B (Weights & Biases):** Best for teams doing deep learning with heavy experimentation.
  The real-time dashboard, system metrics (GPU utilization), and Sweeps (hyperparameter tuning)
  are best-in-class. Trade-off: SaaS dependency and cost at scale.

- **Neptune:** Best for teams needing strong collaboration features and custom metadata.
  Very flexible schema. Trade-off: higher learning curve, SaaS pricing.

- **CometML:** Best for teams wanting a balance between MLflow's openness and W&B's polish.
  Good free tier. Trade-off: smaller community than MLflow or W&B.

### Why we chose MLflow

1. **Open source and self-hosted.** No data leaves our infrastructure. Critical for financial
   data like credit card transactions.
2. **Native XGBoost support.** The `mlflow.xgboost` flavor handles model serialization
   natively.
3. **Excellent DVC integration.** Both tools use a file-based approach that complements
   rather than conflicts.
4. **Industry standard.** Most MLOps job postings list MLflow. It is the most widely
   adopted experiment tracking tool.
5. **Simple local development.** No server needed for single-developer use --- just a
   local `mlruns/` directory.

---

## Advanced MLflow Patterns

### Custom Metrics Logging

Our `train.py` logs cross-validation metrics with step tracking:

```python
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    fold_model = xgb.XGBClassifier(**model_params)
    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
    fold_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_prob)
    cv_scores.append(fold_auc)
    mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)  # Step = fold number
```

This creates a metric history that can be plotted in the MLflow UI as a line chart:

```
cv_auc_roc
  |
  |     *         *
  |  *     *   *
  |
  +--1--2--3--4--5--> fold
```

### Connecting Training Metrics to Evaluation Metrics

Our pipeline writes metrics to two separate JSON files:
- `metrics/train_metrics.json` --- Written by `train.py`, includes the MLflow run ID
- `metrics/eval_metrics.json` --- Written by `evaluate.py`

The train metrics file includes the MLflow run ID:
```python
train_metrics = {
    **{f"val_{k}": v for k, v in val_metrics.items()},
    "cv_auc_roc_mean": float(np.mean(cv_scores)),
    "cv_auc_roc_std": float(np.std(cv_scores)),
    "run_id": run.info.run_id,  # Links back to MLflow
}
```

This creates an audit trail: given `eval_metrics.json`, you can find the MLflow run ID
in `train_metrics.json`, and from there access all parameters, metrics, and artifacts
for that training run.

---

## Interview Questions and Answers

### Q1: Why do you need experiment tracking? Can't you just use a notebook?

**A:** Notebooks are fine for a single person doing a few experiments. They break down
when: (1) You run hundreds of experiments and need to compare them systematically. (2)
Multiple team members are experimenting and need to share results. (3) You need to
reproduce a result from weeks ago and have lost track of which notebook version produced
it. (4) You need an audit trail for regulatory compliance. (5) You want to automate
model selection (e.g., "deploy the model with the highest AUC-ROC"). Experiment tracking
tools provide all of this out of the box.

### Q2: How does MLflow differ from TensorBoard?

**A:** TensorBoard is primarily a visualization tool for training metrics (loss curves,
histograms, computation graphs). It is tightly coupled to TensorFlow/PyTorch. MLflow is
a complete experiment management platform that includes tracking, model packaging, model
registry, and project packaging. MLflow is framework-agnostic --- it works with XGBoost,
scikit-learn, PyTorch, TensorFlow, and any custom model. MLflow also provides model
versioning and stage management (staging, production, archived), which TensorBoard does not.

### Q3: What is the MLflow Model Registry and when would you use it?

**A:** The Model Registry is a centralized catalog of trained models with versioning and
lifecycle management. You would use it when: (1) Multiple team members are training models
and you need to manage which version is in production. (2) You need a formal approval
process before deploying a model (staging -> production transition). (3) You want to
maintain a history of all production models for rollback. (4) You need annotations and
descriptions attached to model versions for documentation. In our project, we use a simpler
approach (GitHub Actions quality gates), but the Model Registry would be the next evolution
as the team grows.

### Q4: How would you handle experiment tracking in a distributed training setup?

**A:** In distributed training, multiple workers train the same model in parallel. The
approach is: (1) Only the rank-0 (master) worker logs to MLflow to avoid duplicate entries.
(2) Use a centralized MLflow tracking server (not local file storage) accessible from all
workers. (3) Aggregate metrics from all workers before logging (e.g., average loss across
workers). (4) Log system metrics (GPU utilization, memory) per worker as separate metrics
with the worker ID as a prefix. (5) Use MLflow's `log_artifact()` to save the final
consolidated model, not per-worker checkpoints.

### Q5: How do you decide which metrics and parameters to log?

**A:** Log everything you might want to compare or reproduce. Specifically: (1) All
hyperparameters, even defaults (they might change across library versions). (2) Data
characteristics (sample count, class balance, feature count). (3) All evaluation metrics
relevant to the business problem (for fraud: precision, recall, F1, AUC-ROC). (4) Training
metadata (duration, hardware, library versions). (5) Data version (DVC hash or commit
SHA). The cost of logging is negligible compared to the cost of not having a metric when
you need it later. In our project, we log 15+ parameters and 9+ metrics per run.

### Q6: How do you prevent "metric gaming" where a model scores well on test data but poorly in production?

**A:** Several strategies: (1) Strict train/val/test splits with no data leakage (our
`preprocess.py` uses stratified splitting). (2) Cross-validation to verify the metric
is stable across folds (our `train.py` runs 5-fold CV). (3) Quality gates that check
multiple metrics, not just one (our gates check recall, precision, F1, AND AUC-ROC). (4)
Holdout test set that is NEVER used during training or hyperparameter tuning (our
`evaluate.py` uses a separate test set). (5) Production monitoring that compares live
performance to offline metrics (our drift detection and performance monitoring). If there
is a large gap between offline and online performance, something is wrong with the
evaluation methodology.

### Q7: What is the relationship between MLflow and DVC in your project?

**A:** They are complementary. DVC handles data versioning and pipeline orchestration
(which data produced which model, in what order). MLflow handles experiment tracking
(what hyperparameters and metrics were associated with each training run). DVC answers:
"Given this data, reproduce this model." MLflow answers: "What was the best performing
configuration across all my experiments?" Together they provide complete reproducibility:
DVC pins the data and pipeline, MLflow pins the experiment details.

### Q8: How would you set up MLflow for a team of 10 data scientists?

**A:** (1) Deploy a centralized MLflow tracking server with a PostgreSQL backend store and
S3 artifact store. (2) Set up authentication (MLflow does not natively support auth, so put
it behind a reverse proxy like Nginx with OAuth/LDAP). (3) Establish naming conventions for
experiments (e.g., `team-name/project-name/experiment-type`). (4) Set up the Model Registry
with approval workflows for production deployments. (5) Create shared dashboards or saved
searches for key metrics. (6) Document logging standards: which metrics to log, naming
conventions for parameters, required tags. (7) Set up garbage collection for old runs
to manage storage costs.

---

## Practical Tips

1. **Log parameters as a dictionary.** Use `mlflow.log_params(params_dict)` instead of
   calling `mlflow.log_param()` individually. It is cleaner and ensures all params are
   logged atomically.

2. **Use `input_example` when logging models.** It documents the expected input format and
   enables MLflow to infer the model signature automatically. Our code passes
   `input_example=X_val.head(1)`.

3. **Include the run ID in your metrics files.** Our `train_metrics.json` includes
   `"run_id": run.info.run_id`. This creates a link between DVC-tracked metrics and
   MLflow-tracked experiments.

4. **Name your runs descriptively.** We use `run_name="xgboost-fraud-detection"`. For
   hyperparameter sweeps, include the key parameters: `run_name=f"xgb-depth{d}-lr{lr}"`.

5. **Do not log inside tight loops.** Logging has I/O overhead. Log aggregated metrics
   (like CV fold AUC) per fold, not per training batch. Our code logs `cv_auc_roc` once
   per fold, not once per gradient step.

6. **Use `cache: false` in DVC for metrics files.** This keeps metrics in Git where
   `dvc metrics diff` can compare them across commits, while MLflow provides the richer
   experiment comparison UI.
