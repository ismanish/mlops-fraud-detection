# 09 — MLOps Interview Questions & Answers

## Table of Contents
1. [MLOps Fundamentals (10 Questions)](#mlops-fundamentals)
2. [Data Management & Versioning (8 Questions)](#data-management--versioning)
3. [Model Training & Experimentation (8 Questions)](#model-training--experimentation)
4. [Model Deployment & Serving (8 Questions)](#model-deployment--serving)
5. [CI/CD for ML (6 Questions)](#cicd-for-ml)
6. [Monitoring & Observability (8 Questions)](#monitoring--observability)
7. [Infrastructure & Cloud (6 Questions)](#infrastructure--cloud)
8. [System Design Scenarios (6 Questions)](#system-design-scenarios)

---

## MLOps Fundamentals

### Q1: What is MLOps and how does it differ from DevOps?

**A:** MLOps extends DevOps principles to machine learning systems, addressing challenges unique to ML: data versioning, experiment tracking, model validation, drift detection, and continuous retraining. While DevOps focuses on CI/CD for deterministic software, MLOps must handle the additional dimension of data and model artifacts. Code changes in traditional software produce predictable outcomes, but ML model quality depends on training data, hyperparameters, and the statistical relationship between features and targets -- all of which can change independently. In our project, we apply DevOps practices (GitHub Actions CI/CD, Docker, Terraform) and layer ML-specific tooling on top (DVC for data versioning, MLflow for experiment tracking, drift detection for production monitoring).

### Q2: Explain the ML lifecycle stages and the tools you used for each.

**A:** The ML lifecycle has five stages in our project. (1) **Data Management**: DVC versions our credit card fraud dataset and tracks data lineage through the pipeline; pandera validates data schemas before training. (2) **Experimentation**: MLflow tracks every training run with hyperparameters, metrics (AUC-ROC, F1, recall), and model artifacts; params.yaml centralizes configuration. (3) **Validation**: Automated quality gates in `evaluate.py` enforce minimum recall (0.80), precision (0.50), and AUC-ROC (0.95) thresholds before any model can be deployed. (4) **Deployment**: GitHub Actions builds a Docker container, pushes to ECR, updates Lambda, and runs a smoke test. (5) **Monitoring**: Custom drift detection using KS tests and PSI compares production distributions against training baselines, with CloudWatch dashboards and alarms.

### Q3: What is the difference between model-centric and data-centric MLOps?

**A:** Model-centric MLOps focuses on improving the model itself -- trying different architectures, hyperparameters, and algorithms while keeping the data fixed. Data-centric MLOps focuses on improving the data -- fixing labels, removing noise, augmenting underrepresented classes, and ensuring data quality. In practice, data-centric approaches often yield larger improvements, especially for production systems. In our project, we take a balanced approach: the model is XGBoost with tuned hyperparameters (model-centric), but we invest heavily in data quality through pandera schema validation, stratified splitting to preserve class distribution, and proper scaling with StandardScaler fit only on training data (data-centric).

### Q4: What are the maturity levels of MLOps?

**A:** Google's MLOps maturity model defines three levels. **Level 0 (Manual)**: Data scientists train models in notebooks, manually deploy to production, no automation -- this is where most companies start. **Level 1 (ML Pipeline Automation)**: Automated training pipelines (like our DVC pipeline), continuous training triggered by data changes or schedules, but deployment is still manual. **Level 2 (CI/CD for ML)**: Full automation including code testing, model validation, automated deployment, and production monitoring with retraining triggers -- this is what our project implements with GitHub Actions workflows for CI (lint, test), CT (train, evaluate, quality gates), and CD (Docker build, ECR push, Lambda update, smoke test).

### Q5: Why is reproducibility important in ML, and how do you achieve it?

**A:** Reproducibility means that given the same data, code, and configuration, you can produce the same model with the same performance. Without it, you cannot debug production issues ("which exact model is running?"), satisfy regulatory audits (financial fraud models require explainability), or confidently roll back to a previous version. Our project achieves reproducibility through: fixed random seeds (`random_state: 42` in params.yaml), data versioning with DVC (every dataset version has a unique hash), experiment tracking with MLflow (every run records hyperparameters, metrics, and the model artifact), Docker images tagged with git commit SHAs (tying each deployment to exact code), and Terraform for infrastructure (ensuring the serving environment is identical across deployments).

### Q6: Explain feature stores and when you would use one.

**A:** A feature store is a centralized repository for storing, serving, and sharing engineered features across ML models and teams. It provides two interfaces: an offline store (for batch training, backed by a data warehouse) and an online store (for real-time serving, backed by a low-latency database like Redis or DynamoDB). Feature stores solve the training-serving skew problem: features computed one way during training and a different way during serving lead to silent model degradation. In our project, we do not use a feature store because we have a single model with simple features (PCA components from the dataset). If we had multiple models sharing features (fraud detection, credit scoring, customer segmentation), or if feature engineering was complex (aggregations over time windows), a feature store like Feast, Tecton, or SageMaker Feature Store would reduce duplication, ensure consistency, and provide point-in-time correctness for training data.

### Q7: What is training-serving skew and how do you prevent it?

**A:** Training-serving skew occurs when the model receives different feature values during serving than it saw during training, due to inconsistencies in feature computation. Common causes include: (1) using different code paths for training and serving (e.g., computing features in Spark for training but Python for serving), (2) data leakage where the scaler or encoder is fit on the entire dataset instead of just training data, (3) different library versions between training and serving environments. Our project prevents skew by: fitting `StandardScaler` only on training data and persisting it to `scaler.pkl` (used identically during serving in `predict.py`), packaging the exact same code and dependencies in a Docker container for both training and serving, and using params.yaml to ensure the same feature columns are dropped in preprocessing and prediction.

### Q8: How do you handle class imbalance in your fraud detection model?

**A:** Our dataset has a 0.17% fraud rate (492 fraud cases out of 284,807 transactions), which is extreme class imbalance. We address this at multiple levels. **At the data level**: stratified train/val/test splitting ensures each split preserves the original class distribution. **At the model level**: XGBoost's `scale_pos_weight: 50` parameter increases the cost of misclassifying minority-class (fraud) samples, effectively oversampling the positive class during gradient computation. **At the evaluation level**: we use threshold-independent metrics (AUC-ROC, Average Precision/PR-AUC) rather than accuracy, and our quality gates prioritize recall (min 0.80) over precision (min 0.50) because missing fraud is costlier than false alarms. **At the threshold level**: in `predict.py`, we use a 0.5 probability threshold, but in production, you would tune this based on the business cost matrix.

### Q9: What are quality gates and why are they critical for ML CI/CD?

**A:** Quality gates are automated checks that a model must pass before it can be deployed. They act as guardrails preventing a degraded model from reaching production. In our project, the `_check_quality_gates()` function in `evaluate.py` enforces four thresholds: recall >= 0.80, precision >= 0.50, F1 >= 0.60, and AUC-ROC >= 0.95. If any threshold is violated, the function raises a `ValueError`, which causes the GitHub Actions workflow to fail, blocking deployment. The `train.yml` workflow has a separate `deploy-gate` job that downloads evaluation metrics and re-checks thresholds as an additional safeguard. Quality gates are critical because ML models can degrade silently -- a retrained model might have a subtle data bug that reduces recall to 0.40, which would go unnoticed without automated checks but would mean 40% of fraud cases are missed.

### Q10: Explain the concept of an ML pipeline vs an ML workflow.

**A:** An ML pipeline is a directed acyclic graph (DAG) of data processing and model training steps where each step's outputs feed into the next step's inputs. Our DVC pipeline defines five stages: `ingest -> validate -> preprocess -> train -> evaluate`, with explicit dependencies (deps), outputs (outs), and parameters (params) for each stage. A workflow is the broader orchestration that may include multiple pipelines plus operational tasks. Our GitHub Actions workflows orchestrate: the CI workflow (lint + test on every push), the training workflow (run the DVC pipeline + quality gates on code changes), and the deployment workflow (build Docker + push to ECR + update Lambda after training succeeds). The pipeline ensures data flows correctly; the workflow ensures the right pipeline runs at the right time with the right triggers.

---

## Data Management & Versioning

### Q11: How does DVC work and why did you choose it over alternatives?

**A:** DVC (Data Version Control) extends git to handle large files. When you run `dvc add data/raw/creditcard.csv`, DVC replaces the file with a small `.dvc` pointer file (containing the MD5 hash), moves the actual data to a local cache (`.dvc/cache/`), and adds the real file to `.gitignore`. The pointer file is committed to git, so git history tracks which data version corresponds to which code version. The actual data can be pushed to remote storage (S3, GCS, Azure Blob) via `dvc push`. We chose DVC over alternatives because: (1) it integrates natively with git (no separate version tracking system), (2) `dvc.yaml` defines reproducible pipelines with dependency tracking, (3) it is open-source and widely adopted, and (4) it supports S3 as a remote, matching our AWS infrastructure. Alternatives like LakeFS (git-like branching for data lakes) or Delta Lake (versioned tables) are better for data warehouse scenarios but overkill for our single-dataset use case.

### Q12: Explain your DVC pipeline and how it ensures reproducibility.

**A:** Our `dvc.yaml` defines five stages with explicit dependencies. For example, the `train` stage declares `deps: [src/models/train.py, data/processed/X_train.csv, ...]`, `params: [model, training]`, and `outs: [models/model.pkl]`. When you run `dvc repro`, DVC checks whether any dependency has changed (by MD5 hash). If nothing changed, the stage is skipped. If `params.yaml`'s model section changed (e.g., `n_estimators: 200` to `300`), DVC reruns `train` and all downstream stages (`evaluate`). This means you can reproduce any historical experiment by checking out the git commit (which has the DVC pointer files and params.yaml) and running `dvc repro` -- DVC will either pull cached outputs from remote storage or rerun the pipeline with identical inputs.

### Q13: How do you validate data quality before training?

**A:** Our `src/data/validate.py` uses pandera to enforce a strict schema on the raw dataset. The schema specifies: (1) exact column names and types (Time: float, V1-V28: float, Amount: float >= 0, Class: float in {0.0, 1.0}), (2) no nullable columns, and (3) coercion enabled to handle minor type mismatches. Beyond schema validation, we perform statistical checks: null value detection, duplicate row counting, and a fraud ratio sanity check (if fraud ratio > 0.5, the data is likely corrupted -- our expected ratio is 0.17%). This validation runs as the second DVC pipeline stage, after ingestion but before preprocessing, ensuring that corrupted or malformed data never reaches the training stage.

### Q14: How do you handle the train/val/test split and why does the order matter?

**A:** We perform a two-stage stratified split. First, we split 80% train+val / 20% test. Then we split the train+val set into 87.5% train / 12.5% validation (which yields 10% of total data for validation). Stratified splitting (`stratify=y`) preserves the 0.17% fraud ratio in each split, which is critical for imbalanced datasets -- a random split could produce a validation set with zero fraud cases. The split order matters because: (1) the test set is held out first and never used during training or hyperparameter tuning, ensuring an unbiased estimate of production performance, (2) the validation set is used for early stopping and model selection, and (3) the scaler is fit only on `X_train` and then applied to val and test, preventing data leakage.

### Q15: What is data leakage and how does your preprocessing prevent it?

**A:** Data leakage occurs when information from the test/validation set influences model training, leading to overly optimistic performance estimates that do not generalize. Our preprocessing prevents leakage in three ways: (1) The StandardScaler for the Amount feature is `fit_transform`ed on `X_train` only, then `transform`ed (not fit) on `X_val` and `X_test` -- this ensures the scaler's mean and standard deviation come exclusively from training data. (2) The scaler object is serialized to `scaler.pkl` and loaded identically during serving, ensuring production predictions use the same transformation. (3) The Time column is dropped entirely during preprocessing, preventing temporal leakage (the model could learn to use Time as a proxy for fraud patterns that evolve chronologically).

### Q16: How would you handle data versioning for a dataset that updates daily?

**A:** For incrementally growing datasets, I would use a combination of DVC and a partitioned storage strategy. Each day's data would land in a date-partitioned S3 path (e.g., `s3://bucket/raw/2026/05/16/transactions.parquet`). DVC would track a manifest file listing all partition paths included in each training dataset version. For retraining, a script would combine the last N days of partitions into a training dataset, validate it, and run `dvc add` to version the combined result. This approach gives you: full audit trail of which data trained which model, the ability to reproduce any historical training run, and incremental data ingestion without re-downloading the entire dataset. For our fraud detection use case, I would retain the last 90 days of data for training, with older data archived to S3 Glacier.

### Q17: Explain the difference between data validation and data testing.

**A:** Data validation checks structural properties of the data at pipeline runtime: correct schema, expected types, no nulls, value ranges (Amount >= 0), and statistical properties (fraud ratio < 0.5). Our pandera schema in `validate.py` performs data validation. Data testing, by contrast, is done during CI and verifies that the data processing code works correctly on sample data. Our `tests/unit/test_preprocess.py` tests that the preprocessing function produces correct output shapes, applies scaling consistently, and handles edge cases. Data validation catches data issues (corrupted files, schema changes), while data testing catches code issues (bugs in feature engineering). Both are necessary: a correct preprocessing function applied to invalid data, or a buggy function applied to valid data, both produce a broken model.

### Q18: How would you handle sensitive data (PII) in your ML pipeline?

**A:** Our dataset already anonymizes sensitive data using PCA -- the V1-V28 features are principal components of the original transaction features, making it impossible to recover the original values. For pipelines with PII, I would implement: (1) data encryption at rest (we already use S3 SSE-AES256) and in transit (HTTPS for all API calls), (2) access controls via IAM policies restricting who can read raw data vs processed/anonymized data, (3) differential privacy techniques during training to prevent model memorization of individual records, (4) data masking in non-production environments (staging/dev get synthetic or masked data), and (5) audit logging of all data access. For GDPR compliance, I would also implement a data deletion pipeline that can remove a specific customer's data and retrain the model without it.

---

## Model Training & Experimentation

### Q19: Walk through your model training code and explain each step.

**A:** Our `train_model()` function in `src/models/train.py` follows a structured flow. First, it loads processed data (X_train, y_train, X_val, y_val) from CSV files created by the preprocessing stage. It initializes an MLflow experiment and starts a run. Within the run, it logs all hyperparameters from params.yaml (n_estimators, max_depth, learning_rate, etc.) and dataset metadata (sample counts, fraud ratio). It trains an XGBClassifier with the validation set for monitoring via `eval_set`. After training, it computes validation metrics (precision, recall, F1, AUC-ROC, average precision) and logs them to MLflow. It then runs 5-fold stratified cross-validation to estimate model variance and logs the mean and standard deviation of AUC-ROC. Finally, it logs the model to MLflow, saves it locally as `model.pkl` via joblib, and writes training metrics to `metrics/train_metrics.json`.

### Q20: Why did you choose XGBoost over other algorithms for fraud detection?

**A:** XGBoost was chosen for several practical reasons. (1) **Performance**: gradient-boosted trees consistently rank among the top performers for tabular data, especially with the structured PCA features in our dataset. (2) **Handling imbalance**: the `scale_pos_weight` parameter natively handles class imbalance without requiring external resampling. (3) **Inference speed**: XGBoost predictions are extremely fast (~1ms for a single sample), critical for real-time fraud detection with strict latency requirements. (4) **Model size**: serialized XGBoost models are compact (~3MB), fitting easily within Lambda's container image limits. (5) **Feature importance**: built-in feature importance scores enable model interpretability. Alternatives considered: LightGBM (similar performance, slightly faster training, but less widespread adoption), random forests (simpler but less accurate), and neural networks (overkill for 29 tabular features, slower inference, harder to deploy on Lambda).

### Q21: Explain your cross-validation strategy and why you used StratifiedKFold.

**A:** We use 5-fold stratified cross-validation on the training set to estimate model generalization and variance. StratifiedKFold ensures each fold preserves the original class distribution (0.17% fraud). With standard KFold on our highly imbalanced dataset, some folds might have zero fraud cases, making AUC-ROC undefined and the CV estimate meaningless. The CV mean AUC-ROC tells us the expected performance on unseen data, while the standard deviation tells us how stable the model is across different data subsets. A CV std > 0.05 would indicate high variance, suggesting overfitting or insufficient training data. We log each fold's AUC-ROC to MLflow as a step metric (`mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)`) for visual inspection of per-fold consistency.

### Q22: How does MLflow tracking work in your project?

**A:** MLflow's tracking server records four categories of information for each training run. **Parameters**: all hyperparameters from params.yaml (n_estimators, max_depth, learning_rate, etc.) plus dataset metadata (train_samples, fraud_ratio, n_features). **Metrics**: validation metrics (val_precision, val_recall, val_f1, val_auc_roc), cross-validation results (cv_auc_roc_mean, cv_auc_roc_std), and per-fold step metrics. **Artifacts**: the serialized XGBoost model with an input example for schema inference. **Run metadata**: run_id, start/end time, status. Our tracking URI is local (`mlruns/` directory), but for team collaboration, you would point it to a remote tracking server or use MLflow on Databricks. The run_id is saved to `metrics/train_metrics.json` and can be used to load any historical model via `mlflow.xgboost.load_model(f"runs:/{run_id}/model")`.

### Q23: What is early stopping and why is it important?

**A:** Early stopping monitors a validation metric during training and stops when the metric stops improving, preventing overfitting. In our XGBoost configuration, we pass `eval_set=[(X_val, y_val)]` so that XGBoost evaluates the model on the validation set after each boosting round. Although our current code does not explicitly set `early_stopping_rounds` in the `fit()` call, `params.yaml` defines `early_stopping_rounds: 20`, meaning training would stop if the validation metric (aucpr -- area under the precision-recall curve) does not improve for 20 consecutive rounds. Without early stopping, XGBoost would train for all 200 rounds even if the model overfits after round 80. Early stopping saves compute time and produces a model that generalizes better, which is especially important for imbalanced datasets where overfitting to the majority class is easy.

### Q24: How do you determine optimal hyperparameters?

**A:** Our current hyperparameters in params.yaml were selected based on domain knowledge and best practices for imbalanced tabular data. `max_depth: 6` prevents overly deep trees that memorize training data. `learning_rate: 0.1` with `n_estimators: 200` balances learning speed with ensemble diversity. `scale_pos_weight: 50` approximately equals the ratio of negative to positive samples (99.83/0.17 ~ 587, but 50 works well empirically to avoid over-correction). `subsample: 0.8` and `colsample_bytree: 0.8` add randomness to each tree, reducing overfitting. For systematic optimization, I would use Optuna or Ray Tune with the cross-validation AUC-ROC as the objective, running a Bayesian hyperparameter search over the space. MLflow would log every trial, and the best configuration would be written back to params.yaml.

### Q25: What is the `eval_metric: aucpr` parameter and why did you choose it?

**A:** `aucpr` is the area under the precision-recall curve, which is the most appropriate metric for highly imbalanced binary classification. Standard AUC-ROC can be misleadingly high for imbalanced datasets because it accounts for true negatives, which dominate when 99.83% of transactions are legitimate. A model that predicts "legitimate" for everything would have ~99.83% accuracy and a decent AUC-ROC. PR-AUC focuses exclusively on the model's performance on the positive class (fraud), making it sensitive to missed fraud cases and false alarms. By optimizing aucpr during training, XGBoost's boosting iterations prioritize improving fraud detection rather than overall classification accuracy. This aligns with our business objective: catching fraud matters more than correctly classifying legitimate transactions.

### Q26: How would you implement A/B testing for model evaluation?

**A:** A/B testing compares two model versions on live production traffic. I would implement it using Lambda aliases with weighted routing: the `production` alias gets 90% of traffic (current model) and the `canary` alias gets 10% (new model). Both models log predictions and confidence scores to CloudWatch with a dimension for model version. After a sufficient sample (e.g., 10,000 predictions per model), I would compare metrics: detection rate, false positive rate, latency, and -- once fraud investigations complete -- precision and recall on labeled outcomes. The statistical test for significance would be a two-proportion z-test for binary metrics or a t-test for continuous metrics like latency. Key consideration for fraud detection: the sample must include enough positive cases (actual fraud) for the comparison to be statistically meaningful, which may require running the test for weeks given the 0.17% fraud rate.

---

## Model Deployment & Serving

### Q27: Explain your deployment pipeline end-to-end.

**A:** Our deployment is a three-workflow GitHub Actions pipeline. (1) **CI (`ci.yml`)**: triggered on every push/PR to main -- runs ruff for linting, black for formatting, mypy for type checking, and pytest for unit and integration tests. (2) **Training (`train.yml`)**: triggered on pushes to main that modify `src/models/`, `src/data/`, or `params.yaml` -- runs the full DVC pipeline (ingest, validate, preprocess, train, evaluate), executes model quality tests, and checks quality gates. (3) **Deployment (`deploy.yml`)**: triggered after training workflow succeeds on main, or manually via workflow_dispatch -- builds a Docker image, pushes to ECR with commit-SHA tag, updates the Lambda function code, waits for the update to complete, and runs a smoke test (health check) against the API Gateway endpoint. Each workflow has clear failure modes: lint errors block testing, quality gate failures block deployment, and smoke test failures alert the team.

### Q28: Why do you use container images for Lambda instead of zip packages?

**A:** Lambda zip packages have a 50MB compressed / 250MB uncompressed limit. Our dependencies alone (XGBoost, scikit-learn, pandas, NumPy, FastAPI, boto3) exceed 300MB uncompressed. Container images support up to 10GB, removing this constraint entirely. Additional benefits: (1) we use the same Docker image for local development (`docker run -p 8080:8080 ...`) and production deployment, eliminating environment discrepancies, (2) the AWS-provided Lambda base images include the Lambda Runtime Interface Client, so our Dockerfile is simple, (3) container images are cached in ECR with layer deduplication, making subsequent pushes fast (only changed layers are uploaded), and (4) vulnerability scanning via ECR's scan-on-push feature automatically checks for CVE in OS packages and dependencies.

### Q29: How do you handle model rollback if a deployment goes wrong?

**A:** Every Docker image pushed to ECR is tagged with the git commit SHA (e.g., `abc123def456`), and ECR retains the last 10 images (via lifecycle policy). To rollback: (1) identify the last known good commit SHA from deployment history, (2) run `aws lambda update-function-code --function-name mlops-fraud-detection-predict --image-uri $ECR_URI:$GOOD_SHA`, (3) verify with a health check. This takes under 60 seconds. For automated rollback, I would enhance the smoke test in `deploy.yml` to run prediction tests (not just health checks), and on failure, automatically update Lambda back to the previous image tag. At the infrastructure level, Terraform state can be rolled back to recreate the entire stack. At the model level, MLflow stores every model artifact with its run_id, so any historical model can be loaded and redeployed.

### Q30: What is the difference between model serving and model inference?

**A:** Model inference is the act of running a trained model on input data to produce predictions -- it is pure computation. Model serving is the infrastructure and systems surrounding inference: receiving requests, validating inputs, routing to the correct model version, running inference, formatting responses, logging predictions, publishing metrics, and handling errors. In our project, inference happens in `predict()` in `src/models/predict.py` (about 10 lines of code). Serving happens in `src/serving/app.py` (the FastAPI application), `lambda_handler.py` (the Mangum adapter), and the entire AWS infrastructure (API Gateway, Lambda, CloudWatch). Most MLOps complexity is in serving, not inference: scaling, latency optimization, A/B testing, canary deployments, and observability are all serving concerns.

### Q31: How does your health check endpoint work and why is it important?

**A:** Our `/health` endpoint returns `{"status": "healthy", "model": "loaded"}` with HTTP 200. It serves multiple purposes: (1) the deploy workflow's smoke test calls it to verify the Lambda function is responsive after code update, (2) it can be used by a load balancer health check if we migrate to ECS, (3) monitoring systems can poll it to detect outages. The health check is simple by design -- it should not perform expensive operations (no database queries, no model inference) because it may be called frequently. In production, I would enhance it to verify that `_model is not None` (confirming the model actually loaded) and return a 503 Service Unavailable if not, enabling the API Gateway to route traffic away from unhealthy instances.

### Q32: How would you implement shadow deployment for a new model?

**A:** Shadow deployment (also called dark launching) runs the new model in parallel with the production model on 100% of live traffic, but only the production model's predictions are returned to the client. The new model's predictions are logged for offline comparison. Implementation: (1) Deploy both model versions to separate Lambda functions, (2) Modify the FastAPI app to call both models on each request, returning only the production model's result but logging both predictions, (3) After collecting sufficient data, compare the shadow model's predictions against the production model and against ground truth labels when they become available. This is safer than A/B testing because no customer is ever exposed to the new model's predictions. The trade-off is increased latency (two model calls per request) and compute cost. For fraud detection specifically, shadow deployment is preferred over A/B testing because you cannot afford to miss fraud cases with an untested model.

### Q33: Explain the request/response lifecycle in your serving architecture.

**A:** A prediction request traverses five layers: (1) **API Gateway** receives the HTTPS POST to `/predict`, handles CORS, logs access, and forwards the event to Lambda. (2) **Lambda** starts the container (cold start) or reuses a warm instance, invoking the handler function. (3) **Mangum** translates the Lambda event (JSON with httpMethod, path, body) into an ASGI scope and calls the FastAPI application. (4) **FastAPI** validates the request body against `TransactionRequest` (28 PCA features + Amount), extracts validated data via `model_dump()`, calls the predict function, measures latency, publishes CloudWatch metrics, and returns a `PredictionResponse`. (5) **Predict** loads model/scaler if not cached, scales the Amount feature, runs XGBoost's `predict_proba()`, applies the 0.5 threshold, and returns the prediction dict. The response travels back through the same layers: FastAPI formats JSON, Mangum wraps it in Lambda's response format, Lambda returns to API Gateway, and the client receives the HTTP response.

### Q34: What are the tradeoffs between real-time and batch inference for fraud detection?

**A:** Real-time inference (what we implement) blocks the transaction until the model responds, enabling immediate fraud prevention. Trade-offs: higher infrastructure complexity, strict latency requirements (<500ms), and cost scales with request volume. Batch inference scores transactions after the fact (e.g., every hour), is simpler and cheaper, but allows fraudulent transactions to complete before detection. For credit card fraud, real-time is strongly preferred because: (1) blocking a fraudulent transaction in progress prevents the loss entirely, (2) the cost of a false positive (declined legitimate transaction) is much lower than a missed fraud (financial loss + customer trust), (3) customers expect immediate transaction decisions. Batch inference is appropriate for less time-sensitive fraud types: insurance claim fraud, account application fraud, or internal audit scoring.

---

## CI/CD for ML

### Q35: How is CI/CD for ML different from CI/CD for traditional software?

**A:** Traditional CI/CD validates code: does it compile, do tests pass, does it deploy successfully. ML CI/CD must additionally validate data and models: does the data conform to the expected schema, does the model meet performance thresholds, is the model better than the current production version. Our CI/CD has three distinct pipelines: code validation (ci.yml -- lint, type check, unit tests), model validation (train.yml -- pipeline execution, quality gates, model quality tests), and deployment (deploy.yml -- containerization, Lambda update, smoke test). The model validation pipeline is the unique ML component -- it is triggered by changes to training code or params.yaml, runs the full DVC pipeline, and enforces metric thresholds before allowing deployment. Traditional CI/CD never needs to ask "is this artifact good enough?" -- ML CI/CD always does.

### Q36: Explain your GitHub Actions workflow trigger strategy.

**A:** We use three trigger patterns. **CI (ci.yml)**: `on: push` to main/develop and `on: pull_request` to main -- runs on every code change to catch issues early. **Training (train.yml)**: `on: push` to main with a path filter (`src/models/**, src/data/**, params.yaml`) -- only retrains when model-relevant code or configuration changes, avoiding unnecessary training on README edits. It also supports `on: workflow_dispatch` for manual retraining (useful when drift is detected). **Deployment (deploy.yml)**: `on: workflow_run` triggered when training completes successfully, plus `workflow_dispatch` for manual deploys. The key design: deployment is conditional (`if: github.event.workflow_run.conclusion == 'success'`), so a failed training run never triggers deployment. This chain (code change -> CI -> training -> quality gate -> deployment) ensures no model reaches production without passing all checks.

### Q37: What are model quality tests and how do they differ from unit tests?

**A:** Unit tests verify code correctness: does the preprocessing function produce the right output shape, does the prediction function return the expected format, does the scaler handle edge cases. Model quality tests verify model behavior: does the model achieve minimum performance metrics, does it handle adversarial inputs gracefully, is it fair across subgroups. In our project, `tests/unit/test_preprocess.py` and `tests/unit/test_model.py` are unit tests; `tests/model/test_model_quality.py` is a model quality test. Model quality tests run after training (they need a trained model) and check properties like "AUC-ROC >= 0.90" and "recall >= 0.70". These complement the quality gates in `evaluate.py` by testing additional behavioral properties like calibration, feature importance stability, and prediction consistency on known inputs.

### Q38: How do you handle the dependency between training and deployment workflows?

**A:** We use GitHub Actions' `workflow_run` trigger, which creates an event-driven dependency. The deployment workflow specifies `on: workflow_run: workflows: ["Train & Evaluate Model"]` with `types: [completed]` and `branches: [main]`. The deployment job has a conditional: `if: github.event.workflow_run.conclusion == 'success'`. This means: (1) deployment only runs after training completes, (2) it only runs if training succeeded (not just completed), (3) it only runs for the main branch (not feature branches). The training workflow itself has a two-job structure: `train` (run pipeline, evaluate model) and `deploy-gate` (download metrics, check thresholds). The deploy-gate job uses `needs: train` and runs only on main (`if: github.ref == 'refs/heads/main'`). This creates a chain of trust: training succeeds -> quality gates pass -> deployment triggers.

### Q39: What is the purpose of the smoke test in your deployment pipeline?

**A:** The smoke test is the final validation step after Lambda code update. It calls the health check endpoint to verify the function is responsive. Specifically: it queries API Gateway to find the API URL, sends a GET request to `/health`, and checks for HTTP 200. If the smoke test fails (non-200 response), the deployment workflow exits with failure, alerting the team. The smoke test catches deployment-level issues that unit tests and model quality tests cannot: misconfigured environment variables, missing files in the Docker image, Lambda memory insufficient for model loading, IAM permission errors, API Gateway routing misconfiguration, and container image compatibility issues. It does not catch model quality issues (those are handled by quality gates), but it ensures the infrastructure is functional.

### Q40: How would you implement canary deployments in your CI/CD pipeline?

**A:** I would modify the deployment workflow to use Lambda aliases and weighted routing. The pipeline would: (1) Deploy the new image to a `canary` Lambda alias (10% traffic), (2) Monitor CloudWatch metrics for 30 minutes (error rate, latency, prediction distribution), (3) If metrics are healthy, shift 100% traffic to the new version by updating the `production` alias, (4) If metrics degrade, rollback by pointing the canary alias back to the production version. Implementation in GitHub Actions would use a long-running job with periodic metric checks via the AWS CLI. The weighted routing is configured via Lambda alias routing: `aws lambda update-alias --function-name predict --name production --routing-config AdditionalVersionWeights={"new-version"=0.1}`. This approach catches production-only issues (data drift, edge cases) that test environments cannot replicate.

---

## Monitoring & Observability

### Q41: What is the difference between monitoring and observability?

**A:** Monitoring is tracking predefined metrics and alerting on known failure modes -- for example, our CloudWatch alarm triggers when Lambda errors exceed 10 in 5 minutes. You know what you are looking for. Observability is the ability to understand the internal state of a system from its external outputs, enabling diagnosis of unknown failure modes. For ML systems, observability means: (1) logs that capture feature values, model version, and prediction details for each request (enabling root cause analysis of specific failures), (2) traces that show the full request lifecycle (API Gateway -> Lambda -> model inference -> CloudWatch publish), (3) metrics at multiple levels (infrastructure, application, and ML). Our project implements monitoring (CloudWatch alarms, drift detection) but could improve observability by adding distributed tracing (AWS X-Ray) and structured logging with prediction context.

### Q42: How do you detect model degradation without ground truth labels?

**A:** In fraud detection, ground truth labels (confirmed fraud from investigations) are delayed by days or weeks. We detect degradation without labels using three proxy signals. (1) **Prediction distribution shift**: if our model suddenly predicts 8% fraud when the baseline is 0.17%, something is wrong -- our `FraudPredicted` CloudWatch metric tracks this. (2) **Data drift**: statistical tests (KS, PSI) on input features detect when the production data diverges from training data, which is a leading indicator of future performance degradation. (3) **Confidence calibration**: tracking the distribution of `fraud_probability` values -- if the model becomes more uncertain (probabilities clustering near 0.5 instead of near 0 and 1), it may be losing discriminative power. Our `FraudProbability` CloudWatch metric enables this analysis. These proxies do not replace ground truth evaluation, but they provide early warning days before labeled data is available.

### Q43: Explain your CloudWatch dashboard and what each widget tells you.

**A:** Our dashboard has six widgets in a 2x3 grid. **Prediction Latency (5-min avg)**: shows model inference time, with spikes indicating cold starts or resource contention. If p99 exceeds 500ms, we investigate. **Fraud Detection Rate (5-min sum)**: count of fraud predictions. A sudden spike could mean a fraud attack or model miscalibration. A drop to zero could mean the model is broken. **Data Drift Detected (1-hr sum)**: binary signal from our drift detection pipeline. Any non-zero value triggers investigation. **Model Degradation (1-hr sum)**: binary signal when model metrics fall below thresholds. **Lambda Invocations (5-min sum)**: traffic volume with separate lines for invocations and errors. High error rate relative to invocations indicates a systemic issue. **Lambda Duration (5-min avg)**: total function execution time including cold starts. A gradual increase might indicate memory pressure or model growth.

### Q44: How would you implement real-time anomaly detection on your model's predictions?

**A:** I would use a statistical process control approach. During the initial production period, establish baseline statistics: mean fraud probability, standard deviation, and the expected positive prediction rate. Then, on a rolling window (e.g., last 1000 predictions), compute the current mean and compare against the baseline using a control chart (CUSUM or EWMA). If the current mean deviates beyond 3 sigma, trigger an alert. Implementation: (1) the FastAPI app already publishes `FraudProbability` to CloudWatch per request, (2) create a CloudWatch math expression that computes the running average over 1000 data points, (3) create an alarm when this running average exceeds the baseline by a configurable margin. For more sophisticated detection, use Amazon CloudWatch Anomaly Detection, which builds an ML model of the metric's expected behavior and alerts on deviations without manually setting thresholds.

### Q45: What metrics would you add if you had more time?

**A:** Five additional metrics I would implement: (1) **Feature-level statistics per request** -- mean, min, max of each input feature, published to CloudWatch to detect input anomalies in real-time (e.g., all-zero features indicating a client bug). (2) **Model confidence histogram** -- distribution of fraud_probability values bucketed into [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0], tracking calibration over time. (3) **Business metrics** -- dollar amount of flagged transactions, estimated savings from prevented fraud (requires integration with the transaction amount). (4) **Data quality metrics** -- null rates, out-of-range values, and schema violations at the API level. (5) **Model version tracking** -- a dimension on all metrics identifying which model version produced each prediction, enabling A/B comparison and rollback decisions.

### Q46: How does your alerting strategy avoid alert fatigue?

**A:** Alert fatigue occurs when too many non-actionable alerts cause the team to ignore all alerts. We prevent this through three design principles: (1) **Severity tiering** -- only P0 (Lambda error rate > 50%) triggers PagerDuty; P1 alerts go to Slack; P2 to email; P3 to dashboards only. (2) **Appropriate thresholds** -- our drift alarm triggers only when drift is detected (based on a statistically rigorous KS test with alpha=0.05), not on arbitrary metric movements. The high-error-rate alarm requires 10+ errors over two consecutive 5-minute periods, filtering out transient issues. (3) **`treat_missing_data = "notBreaching"`** -- prevents false alarms during quiet periods when no requests are being processed. In a mature system, I would also implement alert correlation (grouping related alerts) and automatic remediation (drift alarm triggers retraining pipeline, not a human page).

### Q47: How would you monitor for fairness in your fraud detection model?

**A:** Fairness monitoring ensures the model does not discriminate across protected groups. Since our dataset uses PCA-anonymized features, we cannot directly assess demographic fairness. In a real deployment with access to demographic data, I would monitor: (1) **Equalized odds**: the false positive rate and false negative rate should be similar across demographic groups (e.g., the model should not block transactions from certain regions disproportionately). (2) **Demographic parity**: the positive prediction rate (fraction flagged as fraud) should be similar across groups. (3) **Calibration fairness**: a 70% fraud probability should mean ~70% actual fraud rate for all subgroups. Implementation: add demographic dimensions to CloudWatch metrics and create separate dashboards per subgroup. Evidently AI can generate fairness reports comparing metrics across slices. Automated fairness gates could be added to the quality gate pipeline, blocking deployment if disparity exceeds a threshold.

### Q48: Explain the tradeoff between monitoring frequency and cost.

**A:** Higher monitoring frequency provides faster detection but increases cost. Our current setup: CloudWatch metrics are published per prediction request (real-time, ~$0.30/metric/month), drift detection runs on-demand (batch, near-zero cost), and CloudWatch alarms evaluate every 5 minutes (standard resolution, $0.10/alarm/month). Increasing to high-resolution metrics (1-second granularity) would cost $0.30/metric/month x 3 = $0.90 more and provide faster anomaly detection. Running drift detection hourly instead of on-demand would require a Lambda trigger but would catch drift faster. The optimal balance depends on the cost of missed fraud versus monitoring cost. If a single undetected fraudulent transaction costs $10,000, spending $50/month more on monitoring is trivially justified. For our learning project, the current setup balances insight with cost-effectiveness.

---

## Infrastructure & Cloud

### Q49: Why did you choose a serverless architecture over containers (ECS)?

**A:** Serverless (Lambda) was chosen for four reasons specific to this project's requirements: (1) **Zero cost at zero traffic** -- as a portfolio project, we do not want to pay for idle infrastructure. Lambda charges per request; ECS charges per running hour. (2) **No capacity planning** -- Lambda scales from 0 to thousands of concurrent executions automatically. ECS requires configuring auto-scaling policies, min/max tasks, and scaling metrics. (3) **Reduced operational overhead** -- no operating system patching, no container orchestration, no load balancer management. (4) **Sufficient for our workload** -- our XGBoost model has sub-100ms inference, well within Lambda's 15-minute timeout. If we needed sustained high throughput (>1000 req/s), predictable latency (no cold starts), or GPU inference, ECS or SageMaker would be more appropriate.

### Q50: Explain your IAM permission strategy.

**A:** We follow the principle of least privilege. The Lambda execution role has three permission sets: (1) `AWSLambdaBasicExecutionRole` -- a managed policy allowing only CloudWatch Logs creation (write logs). (2) S3 permissions restricted to our specific bucket ARN and only `GetObject`, `PutObject`, `ListBucket` actions -- the function can read model artifacts and write drift reports but cannot delete objects or access other buckets. (3) CloudWatch `PutMetricData` on resource `*` (metric data is not resource-specific in CloudWatch). The API Gateway has a resource-based policy (`aws_lambda_permission`) allowing only our specific API Gateway to invoke the Lambda function, preventing unauthorized invocations. All IAM resources are tagged with `ManagedBy = "terraform"` to prevent manual modifications that could weaken the security posture.

### Q51: How would you implement multi-region deployment?

**A:** Multi-region deployment serves two purposes: latency reduction (serve users from the nearest region) and disaster recovery. Implementation: (1) Deploy identical infrastructure to us-east-1 and eu-west-1 using Terraform with separate state files per region (pass `aws_region` as a variable). (2) Use Route 53 with latency-based routing to direct users to the nearest API Gateway. (3) Store the model artifact in S3 with cross-region replication so both regions have the latest model. (4) Use a single ECR repository in us-east-1 with cross-region replication to eu-west-1 for Docker images. Challenges specific to ML: ensuring both regions use the same model version (eventual consistency risk), coordinating retraining across regions, and maintaining consistent drift detection baselines. For most use cases, a single region with CloudFront CDN for static content is sufficient.

### Q52: What security measures are in place in your infrastructure?

**A:** Six security measures are implemented: (1) **S3 bucket public access block** -- all four settings (block_public_acls, block_public_policy, ignore_public_acls, restrict_public_buckets) are enabled, ensuring data and models are never publicly accessible. (2) **S3 encryption at rest** -- AES-256 server-side encryption for all stored objects. (3) **ECR image scanning** -- vulnerability scanning on every push detects CVEs in base images and dependencies. (4) **IAM least privilege** -- Lambda role has only the permissions it needs. (5) **API Gateway CORS** -- configured to allow all origins (would be restricted in production). (6) **CloudWatch audit logs** -- API Gateway access logs record every request with IP, timestamp, method, status, and latency for forensic analysis. Missing: HTTPS is provided by API Gateway by default, but we could add WAF for DDoS protection, API key authentication, and VPC endpoints for S3/CloudWatch access to keep traffic within the AWS network.

### Q53: How do you manage costs in your AWS infrastructure?

**A:** Cost management is built into the infrastructure at multiple levels. **S3**: lifecycle rules transition old models to Infrequent Access after 90 days (40% cheaper) and expire old drift reports after 365 days, preventing unbounded storage growth. **ECR**: lifecycle policy keeps only 10 images, capping storage at ~3.5GB. **CloudWatch**: 30-day log retention prevents log storage accumulation (5GB/month x 12 months = 60GB without retention). **Lambda**: pay-per-request pricing means zero cost during development/idle periods; 512MB memory is right-sized for our model (not over-provisioned). **Total estimated cost**: ~$11/month under normal usage. For further optimization: implement reserved concurrency as a cost ceiling, use S3 Intelligent-Tiering for automatic storage class optimization, and set up AWS Budgets with alerts at $20 and $50 thresholds to catch unexpected charges.

### Q54: Compare S3 vs EFS vs EBS for ML artifact storage.

**A:** **S3** (what we use): object storage with unlimited capacity, 99.999999999% durability, accessible from Lambda/ECS/EC2/anywhere via HTTP. Best for model artifacts, datasets, and reports that are written once and read occasionally. Cost: $0.023/GB/month. **EFS**: network file system mountable by multiple EC2/ECS/Lambda instances simultaneously with POSIX file system semantics. Best for shared training data that multiple training instances need to read concurrently, or for Lambda functions that need fast file access (models loaded from EFS are faster than from S3). Cost: $0.30/GB/month (13x more than S3). **EBS**: block storage attached to a single EC2 instance, like a virtual hard drive. Best for training instances that need high-throughput disk I/O during training. Not shareable across instances without snapshots. Cost: $0.10/GB/month. We chose S3 because our model is small (3MB), Lambda natively supports S3 access, and S3 integrates with DVC for data versioning.

---

## System Design Scenarios

### Q55: Design an ML system for fraud detection at scale (10,000 TPS).

**A:** At 10,000 transactions per second, the architecture shifts from serverless to container-based:

```
                     +------------------+
                     |   CloudFront     |
                     |   (CDN + WAF)    |
                     +--------+---------+
                              |
                     +--------v---------+
                     | Network Load     |
                     | Balancer (NLB)   |
                     +--------+---------+
                              |
              +---------------+---------------+
              |               |               |
     +--------v----+  +------v------+  +------v------+
     | ECS Task #1 |  | ECS Task #2 |  | ECS Task #N |
     | (FastAPI +  |  | (FastAPI +  |  | (FastAPI +  |
     |  XGBoost)   |  |  XGBoost)   |  |  XGBoost)   |
     +------+------+  +------+------+  +------+------+
            |                |               |
     +------v----------------v---------------v------+
     |              Feature Store (Redis)            |
     |         Pre-computed feature lookups          |
     +----------------------------------------------+
            |                                |
     +------v------+              +---------v---------+
     | Kinesis      |              | CloudWatch /      |
     | (prediction  |              | Grafana           |
     |  logging)    |              | (monitoring)      |
     +--------------+              +-------------------+
```

Key design decisions: (1) **ECS Fargate** over Lambda to eliminate cold starts and reduce per-request cost at high volume. Auto-scale based on CPU/request count with min 10, max 100 tasks. (2) **Feature Store (Redis)** for sub-millisecond feature lookups -- pre-compute customer aggregates (average transaction amount, transaction frequency) as enrichment features. (3) **Model in memory** -- load once at container startup, serve from memory. XGBoost inference is ~1ms per request. (4) **Kinesis** for prediction logging -- decouple logging from serving to avoid adding latency. Consumer writes to S3 for drift detection. (5) **Circuit breaker pattern** -- if the model errors out, fall back to a rules-based system (simple threshold checks) to prevent blocking all transactions. (6) Estimated cost: 30 ECS tasks x $0.04/hr = $30/day for compute, plus NLB, Redis, and monitoring.

### Q56: Design a retraining pipeline that runs automatically on drift detection.

**A:** The automated retraining pipeline is an event-driven architecture:

```
Drift Detected           Step Functions          Human-in-the-Loop
(CloudWatch Alarm)       (Orchestration)         (Optional)
      |                       |                       |
      v                       v                       v
+-----+------+  +-------------+--------------+  +----+------+
| SNS Topic  |->| 1. Fetch latest data (S3)  |  | Approve   |
| (trigger)  |  | 2. Validate data quality   |  | Deploy?   |
+-----+------+  | 3. Run preprocessing       |  | (Slack/   |
      |          | 4. Train new model          |  |  Email)   |
      |          | 5. Evaluate on test set     |  +----+------+
      |          | 6. Compare vs production    |       |
      |          | 7. A/B test for 24 hours    |-------+
      |          | 8. Promote or rollback      |
      |          +-----------------------------+
```

Key decisions: (1) **Step Functions** for orchestration -- handles retries, error handling, and human approval steps. (2) **Champion-challenger comparison** -- the new model must beat the current production model by a margin (e.g., 1% AUC-ROC improvement) to be promoted, preventing thrashing between similar models. (3) **Human-in-the-loop gate** -- for high-stakes fraud detection, require manual approval before full production rollout. (4) **Shadow deployment** before full deployment -- run the new model on live traffic in parallel for 24 hours, logging predictions without serving them, and compare against ground truth as it arrives. (5) **Automatic rollback** -- if post-deployment metrics degrade within 72 hours, automatically revert to the previous model.

### Q57: Design a monitoring system for 50 ML models in production.

**A:** With 50 models, the challenge shifts from individual monitoring to systematic observability:

```
+------------------+     +------------------+     +------------------+
| Model Registry   |     | Metric Store     |     | Alert Manager    |
| (MLflow)         |     | (Prometheus /    |     | (PagerDuty /     |
|                  |     |  CloudWatch)     |     |  Opsgenie)       |
| - Model metadata |     | - Per-model      |     | - Severity rules |
| - Version history|     |   latency, QPS   |     | - Escalation     |
| - Data lineage   |     | - Drift scores   |     | - On-call roster |
+--------+---------+     | - Perf metrics   |     +--------+---------+
         |               +--------+---------+              |
         |                        |                        |
+--------v------------------------v------------------------v--------+
|                         Grafana Dashboard                          |
| +-------------------+  +-------------------+  +-----------------+ |
| | Fleet Overview    |  | Model Detail      |  | Drift Heatmap   | |
| | (50 models,       |  | (selected model,  |  | (features x     | |
| |  traffic light)   |  |  full metrics)    |  |  models, PSI)   | |
| +-------------------+  +-------------------+  +-----------------+ |
+-------------------------------------------------------------------+
```

Design decisions: (1) **Standardized metric namespace** -- every model publishes metrics in the format `mlops/{team}/{model_name}/{metric}`, enabling cross-model queries. (2) **Fleet dashboard** -- a single view showing all 50 models as green/yellow/red based on automated health scoring (composite of latency SLA, drift score, and performance degradation). (3) **Drift heatmap** -- a matrix showing PSI values for each feature of each model, making it easy to spot correlated drift (e.g., a data pipeline issue affecting multiple models). (4) **Tiered monitoring frequency** -- business-critical models (fraud, pricing) monitored every 5 minutes; lower-priority models (recommendations) monitored hourly. (5) **Centralized retraining scheduler** -- a meta-service that prioritizes which models to retrain based on drift severity and business impact, preventing GPU contention when multiple models need retraining simultaneously.

### Q58: Design a feature pipeline for real-time fraud detection.

**A:** Real-time fraud detection requires two categories of features: point-in-time features (from the current transaction) and aggregated features (computed over the customer's history):

```
Transaction Event
       |
       v
+------+-------+
| Feature      |
| Extraction   |
| Service      |
+------+-------+
       |
  +----+----+
  |         |
  v         v
Point-in-   Aggregated
Time        Features
Features    (Redis/DynamoDB)
|           |
|    +------v-------+
|    | Customer     |
|    | Profile      |
|    | - avg_amount |
|    | - txn_count  |
|    | - last_txn   |
|    +------+-------+
|           |
+-----+-----+
      |
      v
+-----+------+
| Feature    |
| Vector     |
| (combined) |
+-----+------+
      |
      v
+-----+------+
| Model      |
| Inference  |
+-----------+
```

Design: (1) **Point-in-time features** extracted directly from the transaction event: amount, merchant category, time of day, device type. These are available instantly. (2) **Aggregated features** pre-computed in Redis: customer's average transaction amount (30-day rolling window), transaction frequency, time since last transaction, number of distinct merchants. Updated asynchronously via a Kinesis consumer whenever a transaction is processed. (3) **Feature computation service** combines both feature types into a single vector in <10ms. The same service is used for both training (offline mode reads from the data warehouse) and serving (online mode reads from Redis), preventing training-serving skew. (4) **Fallback strategy** -- if Redis is unavailable, use default values for aggregated features (global population averages) and log a degraded-mode metric. The model still makes a prediction, just with less information.

### Q59: You deploy a new fraud model and false positives increase 3x. How do you diagnose and fix this?

**A:** Systematic diagnosis in four steps: (1) **Verify the issue** -- check CloudWatch FraudPredicted metric. Confirm the positive prediction rate increased from baseline (0.17%) to ~0.51%. Check if it correlates with the deployment timestamp. (2) **Compare model versions** -- load both the old and new models, run them on the same test set. Compare precision-recall curves. If the new model has lower precision at the same recall point, the issue is the model itself. If they produce similar results on test data, the issue is data drift in production. (3) **Inspect feature distributions** -- run drift detection comparing production data from the last 24 hours against training data. Look for features with high PSI or KS statistic. If specific features drifted, investigate upstream data pipelines. (4) **Fix** -- if the model is worse, rollback via `aws lambda update-function-code` with the previous image tag (takes 30 seconds). If data drifted, fix the upstream pipeline and retrain on corrected data. If the threshold is wrong, adjust the decision threshold in `predict.py` from 0.5 to a higher value (e.g., 0.7) to reduce false positives at the cost of some recall. Deploy the fix through the standard pipeline, not as a hotfix.

### Q60: Design an ML platform that supports multiple teams deploying models.

**A:** A multi-team ML platform requires standardization, self-service, and governance:

```
+------------------------------------------------------------------+
|                        ML Platform                                |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | Model Registry   |  | Feature Store    |  | Experiment      | |
|  | (MLflow)         |  | (Feast/Tecton)   |  | Tracking        | |
|  | - Model catalog  |  | - Shared features|  | (MLflow)        | |
|  | - Version control|  | - Online/offline |  | - Per-team      | |
|  | - Stage gates    |  | - Access control |  |   experiments   | |
|  +------------------+  +------------------+  +-----------------+ |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | Serving Layer    |  | Training Layer   |  | Monitoring      | |
|  | (K8s + KServe)   |  | (K8s + Kubeflow) |  | (Grafana +      | |
|  | - Auto-scaling   |  | - GPU scheduling |  |  Prometheus)    | |
|  | - A/B testing    |  | - Spot instances |  | - Fleet health  | |
|  | - Canary deploys |  | - Notebooks      |  | - Drift alerts  | |
|  +------------------+  +------------------+  +-----------------+ |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | CI/CD Pipelines  |  | Data Platform    |  | Governance      | |
|  | (GitHub Actions / |  | (S3 + Glue +    |  | - Model cards   | |
|  |  Argo Workflows) |  |  Athena)         |  | - Approval gates| |
|  | - Templated      |  | - Data catalog   |  | - Audit logs    | |
|  |   pipelines      |  | - Quality checks |  | - Compliance    | |
|  +------------------+  +------------------+  +-----------------+ |
+------------------------------------------------------------------+
```

Key design principles: (1) **Standardized interfaces** -- every team deploys models via the same CI/CD template (like our GitHub Actions workflows but parameterized). This ensures consistent quality gates, monitoring, and rollback procedures. (2) **Self-service with guardrails** -- teams can experiment freely in their namespace but must pass platform-wide checks (performance thresholds, fairness audits, security scans) before production deployment. (3) **Shared feature store** -- prevents feature duplication and ensures training-serving consistency. Features developed by the fraud team can be reused by the risk team. (4) **Centralized monitoring** -- the fleet dashboard shows all production models, enabling the platform team to detect cross-model issues (e.g., a data pipeline failure affecting multiple models simultaneously). (5) **Cost attribution** -- Kubernetes namespaces per team with resource quotas and chargeback, making teams responsible for their compute costs. (6) **Model governance** -- model cards documenting purpose, limitations, and fairness assessments; approval gates requiring review from ML platform team and business stakeholders before production deployment.
