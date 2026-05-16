# MLOps: The Complete Overview

## What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning,
DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and
efficiently. It is not a single tool or framework but a discipline that addresses the unique
challenges of putting machine learning models into real-world applications.

Traditional software engineering has well-established practices for shipping code: version
control, CI/CD, testing, monitoring. ML systems need all of that plus additional capabilities
to handle data dependencies, model training reproducibility, experiment tracking, model
versioning, data drift detection, and automated retraining. MLOps fills that gap.

```
Traditional Software               ML Systems
+------------------+              +------------------+
|   Code           |              |   Code           |
|   Tests          |              |   Tests          |
|   CI/CD          |              |   CI/CD          |
|   Monitoring     |              |   Monitoring     |
+------------------+              |   + Data         |
                                  |   + Models       |
                                  |   + Experiments  |
                                  |   + Features     |
                                  |   + Drift        |
                                  +------------------+
```

In the real world, Google famously noted that ML code is only a small fraction of a real ML
system. The surrounding infrastructure --- data collection, feature extraction, configuration,
serving infrastructure, monitoring --- dwarfs the model training code itself. MLOps is the
discipline that manages all of this surrounding infrastructure.

---

## Why MLOps Matters

Without MLOps, ML projects suffer from a predictable set of failures:

1. **"It works on my laptop" syndrome.** A data scientist trains a model in a Jupyter notebook
   with specific library versions, specific data slices, and specific random seeds. Reproducing
   the result on another machine or six months later becomes impossible.

2. **Model rot / staleness.** A model deployed today will degrade over time as the real world
   changes. Without monitoring and retraining pipelines, the model silently becomes worse.

3. **No audit trail.** Regulators ask: "Why did the model flag this transaction as fraud?"
   Without experiment tracking and data versioning, there is no defensible answer.

4. **Slow iteration.** Without automated pipelines, deploying a new model version takes days
   of manual work: retraining, testing, packaging, deploying, verifying.

5. **Collaboration breakdown.** Data scientists, ML engineers, and DevOps engineers work in
   silos. No shared language, no shared tooling, no shared process.

### Why it matters for this project specifically

Our fraud detection system processes financial transactions in real time. A bad model can
either miss fraud (costing the bank money) or falsely flag legitimate transactions (costing
the bank customers). The stakes are high, and MLOps practices ensure we can:

- Reproduce any past model (DVC tracks data, MLflow tracks experiments)
- Automatically validate data quality before training (pandera schema in `src/data/validate.py`)
- Enforce quality gates before deployment (thresholds in `params.yaml`)
- Detect when the model starts degrading (drift detection in `src/monitoring/drift_detection.py`)
- Deploy new versions with zero downtime (Lambda + ECR container updates)

---

## MLOps vs DevOps vs DataOps

| Dimension       | DevOps                   | DataOps                      | MLOps                                  |
|-----------------|--------------------------|------------------------------|----------------------------------------|
| **What ships**  | Application code         | Data pipelines + data quality| Models + data + code                   |
| **Versioning**  | Code (Git)               | Data schemas, pipelines      | Code + Data + Models + Params          |
| **Testing**     | Unit, integration, E2E   | Data quality, schema tests   | All of DevOps + model quality tests    |
| **CI/CD**       | Build, test, deploy code | Build, test, deploy pipelines| Train, validate, evaluate, deploy model|
| **Monitoring**  | Uptime, latency, errors  | Data freshness, completeness | All of DevOps + data drift + model perf|
| **Artifacts**   | Binaries, containers     | Datasets, transformations    | Models, datasets, scalers, metrics     |
| **Rollback**    | Revert code deployment   | Revert pipeline version      | Revert model + data version            |

Key insight: MLOps is a superset. It includes all DevOps practices (CI/CD, containerization,
monitoring) and adds ML-specific concerns on top. Our project demonstrates this --- we have
standard DevOps (Docker, GitHub Actions, AWS Lambda) combined with ML-specific tooling (DVC,
MLflow, quality gates, drift detection).

---

## MLOps Maturity Levels (0 through 4)

Google and Microsoft have both published MLOps maturity models. Here is a consolidated view
with where our project sits at each level:

### Level 0: No MLOps (Manual Everything)

```
Data Scientist's Laptop
+-------------------------------------------+
|  Jupyter Notebook                         |
|  - Load data from local CSV              |
|  - Train model                           |
|  - "It works!" (on this machine)         |
|  - Email model.pkl to engineering team   |
+-------------------------------------------+
```

- Training is manual and ad hoc
- No version control for data or models
- No automated testing
- Deployment is "throw it over the wall"
- No monitoring in production

### Level 1: DevOps but No MLOps (Automated Deployment)

- Code is in Git and has CI/CD
- Model is deployed via a pipeline
- But data is not versioned
- No experiment tracking
- No model quality gates
- No monitoring for model performance or drift

### Level 2: Automated Training Pipeline (Where Our Project Lives)

```
+----------+    +-----------+    +-------------+    +--------+    +----------+
|  Ingest  |--->| Validate  |--->| Preprocess  |--->| Train  |--->| Evaluate |
| (DVC)    |    | (pandera) |    | (sklearn)   |    | (xgb)  |    | (gates)  |
+----------+    +-----------+    +-------------+    +--------+    +----------+
     |                                                   |             |
     v                                                   v             v
  S3 Remote                                          MLflow         Quality
  (data                                           (tracking)       Gates
  versioning)                                                    (pass/fail)
                                                                     |
                                              +----------------------+
                                              |
                                              v
                                    +-------------------+
                                    | Deploy to Lambda  |
                                    | (Docker + ECR)    |
                                    +-------------------+
                                              |
                                              v
                                    +-------------------+
                                    | Monitor           |
                                    | (drift + perf)    |
                                    +-------------------+
```

- Automated, reproducible training pipeline (our `dvc.yaml` defines 5 stages)
- Data versioning with DVC + S3 remote
- Experiment tracking with MLflow
- Quality gates prevent bad models from deploying
- Automated deployment via GitHub Actions
- Monitoring for drift and performance degradation

### Level 3: Automated Retraining

Everything in Level 2, plus:
- Drift detection automatically triggers retraining
- Model registry manages model lifecycle (staging -> production)
- A/B testing or shadow deployments for new models
- Automated rollback if the new model is worse

Our project has the building blocks for Level 3 (drift detection publishes CloudWatch alarms,
and the training workflow supports `workflow_dispatch` for on-demand retraining), but does
not yet have a fully automated "drift detected -> retrain -> deploy" closed loop.

### Level 4: Full ML Platform

Everything in Level 3, plus:
- Feature store for shared feature engineering
- Centralized model registry serving multiple teams
- Automated A/B testing with statistical significance
- Self-healing pipelines
- Cost optimization (auto-scaling, spot instances)
- Multi-model serving and routing

---

## The ML Lifecycle

Every ML system follows this lifecycle, whether explicitly managed or not. The difference
MLOps makes is that each phase is automated, versioned, and monitored.

```
   +---> DATA ---------> TRAINING ---------> DEPLOYMENT ---------> MONITORING ---+
   |     - Ingest         - Feature eng      - Containerize        - Drift det   |
   |     - Validate       - Train model      - Push to registry    - Perf mon    |
   |     - Version        - Track exps       - Update service      - Alerting    |
   |     - Store          - Evaluate         - Smoke test          - Logging     |
   |                      - Quality gates                                        |
   |                                                                             |
   +------- RETRAINING (triggered by drift or schedule) <------------------------+
```

### How our project implements each phase:

**1. Data Phase**
- `src/data/ingest.py` --- Downloads or generates synthetic fraud data (284,807 transactions)
- `src/data/validate.py` --- Schema validation with pandera (checks types, ranges, class values)
- `dvc.yaml` --- Tracks data lineage through `ingest -> validate -> preprocess` stages
- S3 remote --- DVC pushes data to `s3://mlops-fraud-detection-011015903780`

**2. Training Phase**
- `src/data/preprocess.py` --- Feature engineering (scaling, train/val/test split, stratification)
- `src/models/train.py` --- XGBoost training with 5-fold CV, MLflow experiment tracking
- `src/models/evaluate.py` --- Holdout evaluation + quality gates (min recall 0.80, min AUC 0.95)
- `params.yaml` --- Single source of truth for all hyperparameters

**3. Deployment Phase**
- `Dockerfile` --- Container image based on AWS Lambda Python 3.12 runtime
- `.github/workflows/deploy.yml` --- Build, push to ECR, update Lambda function
- `scripts/deploy.sh` --- ECR push and Lambda update scripts
- Smoke test --- Health check against API Gateway after deployment

**4. Monitoring Phase**
- `src/monitoring/drift_detection.py` --- KS test + PSI for feature drift detection
- `src/monitoring/performance.py` --- Tracks recall, precision, F1 against thresholds
- `src/serving/app.py` --- Publishes latency and prediction metrics to CloudWatch
- `infrastructure/terraform/cloudwatch.tf` --- Dashboard and alarms

---

## Key Principles and How This Project Implements Them

### 1. Reproducibility

**Principle:** Any experiment, any model, any result should be reproducible by anyone at any time.

**Our implementation:**
- `params.yaml` stores all hyperparameters in one file. Change one value and the entire
  pipeline uses it. No hardcoded magic numbers scattered across notebooks.
- `dvc.yaml` defines the exact dependency graph: which scripts depend on which data files
  and parameters. Running `dvc repro` will reproduce the entire pipeline.
- `random_state: 42` is set in params and propagated to train_test_split and XGBoost.
- DVC tracks data versions. Even if the training data changes, we can `dvc checkout` to
  any previous version.
- MLflow records every parameter, metric, and artifact for every training run.

### 2. Automation

**Principle:** Humans should make decisions, not run scripts. Everything else should be automated.

**Our implementation:**
- `Makefile` --- One command for any operation: `make train`, `make deploy`, `make test`
- `.github/workflows/ci.yml` --- Automatic lint + test on every push and PR
- `.github/workflows/train.yml` --- Automatic retraining when model code or data code changes
- `.github/workflows/deploy.yml` --- Automatic deployment after successful training
- `dvc repro` --- Reruns only stages whose inputs have changed (smart caching)

### 3. Versioning (Code + Data + Models + Parameters)

**Principle:** Version everything. Code alone is not enough to reproduce an ML result.

**Our implementation:**
- **Code:** Git (standard)
- **Data:** DVC tracks `data/raw/creditcard.csv` and all processed files. The `.dvc` metadata
  files live in Git; the actual data lives in S3.
- **Models:** `models/model.pkl` is tracked by DVC and also logged as an MLflow artifact
- **Parameters:** `params.yaml` is in Git. DVC `params:` sections declare which parameters
  each stage depends on.
- **Metrics:** `metrics/eval_metrics.json` is tracked by DVC with `cache: false` so it is
  always committed to Git for easy comparison via `dvc metrics diff`.

### 4. Testing and Quality Gates

**Principle:** Do not deploy a model that has not been rigorously tested. ML models need
quality tests beyond traditional unit/integration tests.

**Our implementation:**
- `tests/unit/` --- Test preprocessing logic and model output format
- `tests/integration/` --- Test the full ingest-validate pipeline
- `tests/model/test_model_quality.py` --- Assert AUC-ROC >= 0.90, recall >= 0.70, precision >= 0.30
- `src/models/evaluate.py::_check_quality_gates()` --- Enforced during evaluation; raises
  ValueError if any threshold is not met
- `train.yml::deploy-gate` job --- GitHub Actions checks metrics before allowing deployment

### 5. Monitoring

**Principle:** A deployed model is not "done." It will degrade. You must watch it continuously.

**Our implementation:**
- `src/serving/app.py::_publish_metrics()` --- Every prediction publishes latency, prediction,
  and fraud probability to CloudWatch
- `src/monitoring/drift_detection.py` --- KS test and PSI compare production data against
  training reference distribution
- `src/monitoring/performance.py` --- Checks if live metrics breach configured thresholds
- `infrastructure/terraform/cloudwatch.tf` --- Dashboard with 6 widgets + 2 alarms
  (high error rate, drift detected)

### 6. Collaboration

**Principle:** Data scientists, ML engineers, and DevOps engineers should be able to work
together using shared tools and processes.

**Our implementation:**
- `params.yaml` --- Everyone reads the same config. A data scientist tweaks hyperparameters;
  an ML engineer adjusts the pipeline; a DevOps engineer manages AWS config. All in one file.
- `.pre-commit-config.yaml` --- Enforces code quality standards for everyone (ruff, trailing
  whitespace, no private keys, no large files)
- `Makefile` --- Common vocabulary: `make train`, `make test`, `make deploy`
- `mlflow ui` --- Shared experiment dashboard accessible to the entire team

---

## Project Architecture

```
mlops-fraud-detection/
|
|-- src/
|   |-- data/
|   |   |-- ingest.py          # Download / generate fraud dataset
|   |   |-- validate.py        # Schema validation with pandera
|   |   |-- preprocess.py      # Feature engineering + train/val/test split
|   |-- models/
|   |   |-- train.py           # XGBoost training + MLflow tracking
|   |   |-- evaluate.py        # Holdout evaluation + quality gates
|   |   |-- predict.py         # Inference module (loads model + scaler)
|   |-- serving/
|   |   |-- app.py             # FastAPI application
|   |   |-- lambda_handler.py  # AWS Lambda handler (Mangum wrapper)
|   |-- monitoring/
|   |   |-- drift_detection.py # KS test + PSI drift detection
|   |   |-- performance.py     # Production metric monitoring
|   |-- utils/
|       |-- config.py          # params.yaml loader + project root
|       |-- logger.py          # Structured logging
|
|-- tests/
|   |-- unit/                  # Preprocessing + model output tests
|   |-- integration/           # End-to-end pipeline tests
|   |-- model/                 # Model quality threshold tests
|
|-- .github/workflows/
|   |-- ci.yml                 # Lint + test on every push/PR
|   |-- train.yml              # Train + evaluate + quality gate
|   |-- deploy.yml             # Build container + push ECR + update Lambda
|
|-- infrastructure/terraform/  # S3, ECR, Lambda, API Gateway, CloudWatch
|-- scripts/                   # deploy.sh, setup_aws.sh, run_pipeline.sh
|-- params.yaml                # All hyperparameters and configuration
|-- dvc.yaml                   # Pipeline DAG definition
|-- Dockerfile                 # Lambda container image
|-- Makefile                   # Developer commands
|-- requirements.txt           # Python dependencies
|-- pyproject.toml             # Project metadata + tool config
|-- .pre-commit-config.yaml    # Pre-commit hooks
```

---

## End-to-End Data Flow

```
                              +--------- Git ----------+
                              |  Code, params.yaml,    |
                              |  dvc.yaml, .dvc files, |
                              |  metrics/*.json        |
                              +------------------------+

Kaggle / Synthetic -----> data/raw/creditcard.csv -----> data/processed/
    (ingest.py)                (validate.py)               (preprocess.py)
                                                              |
                     +------------ S3 (DVC Remote) ---------- | -----+
                     |  data/raw/creditcard.csv               |      |
                     |  data/processed/X_train.csv            |      |
                     |  data/processed/scaler.pkl             |      |
                     |  models/model.pkl                      |      |
                     +----------------------------------------+      |
                                                                     |
                     X_train, y_train, X_val, y_val                  |
                              |                                      |
                              v                                      |
                     +------------------+                            |
                     | train.py         |                            |
                     | (XGBoost + MLflow|                            |
                     |  tracking)       |                            |
                     +------------------+                            |
                              |                                      |
                     models/model.pkl                                |
                              |                                      |
                              v                                      |
                     +------------------+                            |
                     | evaluate.py      |                            |
                     | (quality gates)  |                            |
                     +------------------+                            |
                              |                                      |
                     metrics/eval_metrics.json                       |
                              |                                      |
                 +------------|-------------+                        |
                 | PASS                FAIL |                        |
                 v                     v    |                        |
         +---------------+     Block deploy|                        |
         | Dockerfile    |                 |                        |
         | (build image) |                 |                        |
         +---------------+                                          |
                 |                                                   |
                 v                                                   |
         +---------------+                                          |
         | ECR           |                                          |
         | (push image)  |                                          |
         +---------------+                                          |
                 |                                                   |
                 v                                                   |
    +---------------------------+                                   |
    | AWS Lambda                |                                   |
    |  + API Gateway            |                                   |
    |  + CloudWatch Monitoring  |                                   |
    +---------------------------+                                   |
                 |                                                   |
                 v                                                   |
    +---------------------------+                                   |
    | Drift Detection           | ---------> triggers retraining ---+
    | Performance Monitoring    |
    +---------------------------+
```

---

## Interview Questions and Answers

### Q1: What is MLOps and why is it needed?

**A:** MLOps is a discipline that applies DevOps principles to machine learning systems. It is
needed because ML systems have unique challenges beyond traditional software: data dependencies,
model training reproducibility, experiment tracking, model versioning, data drift, and concept
drift. Without MLOps, teams cannot reliably reproduce results, detect model degradation, or
deploy new models safely. Google found that ML code is typically less than 5% of a production
ML system --- the remaining 95% is infrastructure that MLOps manages.

### Q2: What are the key differences between MLOps and DevOps?

**A:** DevOps versions and deploys code. MLOps versions and deploys code PLUS data, models,
parameters, and metrics. In DevOps, a build either passes tests or fails. In MLOps, a model
passes tests AND must meet quality thresholds (like AUC-ROC >= 0.95). DevOps monitoring
tracks uptime and latency. MLOps monitoring also tracks data drift, prediction drift, and
model performance metrics over time. DevOps CI/CD is triggered by code changes. MLOps CI/CD
can also be triggered by data changes, scheduled retraining, or drift detection alerts.

### Q3: Explain MLOps maturity levels. Where would you place a typical startup?

**A:** There are 5 levels (0-4). Level 0 is fully manual: Jupyter notebooks, emailed model
files, no versioning. Level 1 adds basic DevOps but no ML-specific tooling. Level 2 adds
automated training pipelines, experiment tracking, and data versioning. Level 3 adds automated
retraining triggered by monitoring. Level 4 is a full ML platform with feature stores, model
registries, A/B testing, and multi-team support. Most startups are at Level 0 or 1. Well-run
ML teams at mid-size companies are typically at Level 2. Large tech companies like Google,
Netflix, and Uber operate at Level 3-4.

### Q4: How do you ensure reproducibility in an ML pipeline?

**A:** Reproducibility requires versioning four things: code (Git), data (DVC with S3 remote),
parameters (params.yaml checked into Git), and the execution environment (Docker container
with pinned dependency versions in requirements.txt). Additionally, random seeds must be
fixed and propagated (we use `random_state: 42` in params.yaml). The pipeline DAG (dvc.yaml)
captures the exact sequence of steps and their dependencies. MLflow logs every experiment
run with its parameters, metrics, and artifacts. Together, these tools allow any team member
to reproduce any past experiment exactly.

### Q5: What is the role of quality gates in MLOps?

**A:** Quality gates are automated checks that prevent a model from being deployed if it does
not meet minimum performance thresholds. In our project, the evaluate step checks recall >= 0.80,
precision >= 0.50, F1 >= 0.60, and AUC-ROC >= 0.95. If any threshold is not met, the pipeline
fails and deployment is blocked. This is critical in production because a model that appears
to "work" might actually be worse than the current production model. Quality gates codify the
team's minimum acceptable performance into an automated check, removing the human judgment
bottleneck while ensuring no bad model ever reaches production.

### Q6: How would you handle a situation where your production model starts performing worse?

**A:** First, I would detect the degradation through monitoring --- specifically drift detection
(KS tests and PSI on feature distributions) and performance monitoring (tracking precision,
recall, and F1 against ground truth labels as they become available). When degradation is
detected, I would: (1) Diagnose whether it is data drift (input distribution changed) or
concept drift (the relationship between features and target changed). (2) If data drift,
retrain on recent data. If concept drift, potentially re-engineer features. (3) Run the new
model through the full quality gate pipeline before deploying. (4) Use canary or shadow
deployment to validate the new model in production before full rollover. (5) Set up automated
alerts so this is caught early next time --- which is exactly what our CloudWatch alarms do.

### Q7: Walk me through how you would deploy a new model version in your project.

**A:** The flow is: (1) Make changes to model code or hyperparameters in params.yaml.
(2) Push to main branch, which triggers the `train.yml` workflow. (3) GitHub Actions runs
the full pipeline: ingest, validate, preprocess, train, evaluate. (4) Model quality tests
run (`tests/model/test_model_quality.py`) to assert thresholds. (5) The deploy-gate job
downloads metrics and checks AUC-ROC >= 0.90 and recall >= 0.70. (6) If gates pass, the
`deploy.yml` workflow triggers: build Docker image, push to ECR, update Lambda function.
(7) A smoke test hits the `/health` endpoint via API Gateway to verify the deployment.
(8) CloudWatch monitors the new model's predictions and latency in production.

### Q8: What metrics would you monitor for a fraud detection model in production?

**A:** I would monitor: (1) **Prediction latency** --- fraud detection is real-time, so
latency must stay under 100ms. (2) **Fraud prediction rate** --- a sudden spike or drop
indicates something changed. (3) **Precision and recall** --- recall is especially critical
for fraud because missing a fraud is more costly than a false positive. (4) **Data drift** ---
are the feature distributions changing? We use KS tests and PSI. (5) **Prediction drift** ---
is the model's output distribution changing even if inputs look similar? (6) **Error rate** ---
Lambda invocation errors. (7) **Business metrics** --- false positive rate (customer friction),
actual fraud caught, fraud losses. Our CloudWatch dashboard tracks prediction latency, fraud
detection rate, data drift, model degradation, Lambda invocations, and Lambda duration.

---

## Practical Tips

1. **Start with params.yaml.** Put every configurable value in one YAML file from day one.
   Hyperparameters, file paths, thresholds, AWS config. It becomes the single source of truth.

2. **Make your pipeline idempotent.** Every stage should produce the same output given the
   same input. Our ingest step checks if data already exists before downloading. DVC skips
   unchanged stages automatically.

3. **Separate concerns.** Our `src/` is organized by function: data, models, serving,
   monitoring, utils. Each module does one thing. This makes testing, debugging, and
   onboarding much easier.

4. **Test at three levels.** Unit tests for individual functions. Integration tests for
   pipeline stages. Model quality tests for ML-specific assertions. Run unit and integration
   tests on every PR (ci.yml). Run model quality tests after training (train.yml).

5. **Version your infrastructure.** Our Terraform files in `infrastructure/terraform/` define
   all AWS resources as code. No clicking around in the console.

6. **Use pre-commit hooks.** Our `.pre-commit-config.yaml` catches code quality issues,
   large files, and accidentally committed secrets before they ever reach the repository.
