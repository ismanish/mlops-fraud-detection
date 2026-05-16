# CI/CD for ML with GitHub Actions

## Why CI/CD is Different for ML

Traditional CI/CD is straightforward: code changes trigger a build, tests run, and if
they pass, the artifact is deployed. ML CI/CD is fundamentally more complex because:

1. **Two types of changes matter.** In traditional software, only code changes. In ML,
   both code changes AND data changes can break the system. A new batch of training data
   might shift the distribution, degrading model quality.

2. **Tests are probabilistic.** A unit test in traditional software is deterministic: it
   passes or fails. A model quality test is statistical: "Is AUC-ROC >= 0.95?" The threshold
   is a judgment call, not a logical assertion.

3. **Build time is long.** Compiling code takes seconds or minutes. Training a model can
   take hours or days. This changes how you structure your pipeline.

4. **Artifacts are large.** A compiled binary is typically a few MB. A trained model with
   its data can be gigabytes.

5. **Deployment requires validation beyond tests.** Even if tests pass, you need to verify
   the model is not worse than the current production model. This is the "quality gate"
   concept.

```
Traditional CI/CD:
  Code Change --> Lint --> Test --> Build --> Deploy
  (minutes)

ML CI/CD:
  Code/Data Change --> Lint --> Unit Test --> Ingest --> Validate Data -->
  Preprocess --> Train Model --> Evaluate --> Quality Gate --> Build Container -->
  Push to Registry --> Deploy --> Smoke Test --> Monitor
  (potentially hours)
```

---

## Our 3 Workflows Explained

Our project uses three GitHub Actions workflows, each serving a distinct purpose:

```
                    +--------------------+
                    |  Developer pushes  |
                    |  to branch         |
                    +--------------------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +--------+--+  +--------+--+  +--------+--+
     | ci.yml    |  | train.yml |  | deploy.yml|
     | (every    |  | (main +   |  | (after   |
     |  push/PR) |  |  code     |  |  train   |
     |           |  |  change)  |  |  success)|
     +-----------+  +-----------+  +-----------+
     | Lint      |  | Ingest    |  | Build    |
     | Unit test |  | Validate  |  | Push ECR |
     | Int. test |  | Preprocess|  | Update   |
     |           |  | Train     |  |  Lambda  |
     |           |  | Evaluate  |  | Smoke    |
     |           |  | Quality   |  |  test    |
     |           |  |  gate     |  |          |
     +-----------+  +-----------+  +-----------+
```

### Workflow 1: ci.yml --- Lint and Test

**File:** `.github/workflows/ci.yml`

**Triggers:**
- Every push to `main` or `develop`
- Every pull request targeting `main`

**Purpose:** Fast feedback on code quality. This runs in under 2 minutes and catches
obvious problems before any expensive training happens.

```yaml
name: CI --- Lint & Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install ruff black mypy
      - name: Ruff
        run: ruff check src/ tests/
      - name: Black
        run: black --check src/ tests/
      - name: Mypy
        run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint                    # Tests only run if lint passes
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Unit tests
        run: pytest tests/unit/ -m unit -v --tb=short
      - name: Integration tests
        run: pytest tests/integration/ -m integration -v --tb=short
```

**Key design decisions:**

1. **Lint runs first and fast.** The `lint` job installs only 3 packages (ruff, black, mypy)
   instead of the full requirements.txt. This makes it fast. If lint fails, tests are skipped
   (saving time and compute).

2. **`needs: lint` creates a dependency.** The `test` job waits for `lint` to succeed. There
   is no point running tests if the code has syntax errors.

3. **Three linters for three purposes:**
   - `ruff check` --- Fast Python linter (style errors, import ordering, unused variables)
   - `black --check` --- Code formatter verification (does not modify, just checks)
   - `mypy` --- Static type checking (catches type mismatches, missing attributes)

4. **Two test tiers in the same job:** Unit tests run first (fast, no data dependencies).
   Integration tests run second (create synthetic data, run through the pipeline).

### Workflow 2: train.yml --- Train and Evaluate

**File:** `.github/workflows/train.yml`

**Triggers:**
- Manual dispatch (`workflow_dispatch`) with a "reason" input
- Push to `main` when files in `src/models/**`, `src/data/**`, or `params.yaml` change

**Purpose:** Run the full ML pipeline and validate model quality before deployment.

```yaml
name: Train & Evaluate Model

on:
  workflow_dispatch:
    inputs:
      reason:
        description: "Reason for retraining"
        required: true
        default: "Scheduled retrain"
  push:
    branches: [main]
    paths:
      - "src/models/**"
      - "src/data/**"
      - "params.yaml"
```

**Key design decisions:**

1. **Path-based triggers.** The workflow only runs when ML-relevant files change. Editing
   README.md, Terraform configs, or test files does not trigger a retrain. This saves
   significant compute cost.

2. **`workflow_dispatch` for manual retraining.** Operations teams can trigger a retrain
   from the GitHub UI with a documented reason. This supports both automated and manual
   retraining workflows.

3. **Two jobs: `train` and `deploy-gate`.**

**Job 1: train**
```yaml
  train:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run data ingestion
        run: python -m src.data.ingest
      - name: Run data validation
        run: python -m src.data.validate
      - name: Run preprocessing
        run: python -m src.data.preprocess
      - name: Train model
        run: python -m src.models.train
      - name: Evaluate model
        run: python -m src.models.evaluate
      - name: Run model quality tests
        run: pytest tests/model/ -m model -v --tb=short
      - name: Upload metrics
        uses: actions/upload-artifact@v4
        with:
          name: model-metrics
          path: metrics/
      - name: Upload model
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: models/model.pkl
```

This job runs the complete pipeline sequentially: ingest -> validate -> preprocess ->
train -> evaluate -> quality tests. If any step fails, the pipeline stops. The model
and metrics are uploaded as GitHub Actions artifacts for the next job.

**Job 2: deploy-gate**
```yaml
  deploy-gate:
    runs-on: ubuntu-latest
    needs: train
    if: github.ref == 'refs/heads/main'    # Only on main branch
    steps:
      - name: Download metrics
        uses: actions/download-artifact@v4
        with:
          name: model-metrics
          path: metrics/
      - name: Check quality gates
        run: |
          python3 -c "
          import json
          with open('metrics/eval_metrics.json') as f:
              m = json.load(f)
          print(f'AUC-ROC: {m[\"auc_roc\"]:.4f}')
          print(f'F1: {m[\"f1\"]:.4f}')
          print(f'Recall: {m[\"recall\"]:.4f}')
          assert m['auc_roc'] >= 0.90, f'AUC-ROC too low: {m[\"auc_roc\"]}'
          assert m['recall'] >= 0.70, f'Recall too low: {m[\"recall\"]}'
          print('Quality gates passed --- ready for deployment')
          "
```

The deploy-gate is a separate job that:
- Only runs on the main branch (feature branches do not trigger deployment)
- Downloads metrics from the previous job
- Checks AUC-ROC >= 0.90 and recall >= 0.70
- If assertions fail, the job fails and deployment is blocked

Note the two layers of quality gates:
1. `src/models/evaluate.py::_check_quality_gates()` checks against `params.yaml` thresholds
   (min_recall: 0.80, min_auc_roc: 0.95) --- these are the strict thresholds
2. `train.yml::deploy-gate` checks with slightly relaxed thresholds (AUC-ROC >= 0.90,
   recall >= 0.70) --- these are the deployment floor

The first layer catches bad models during evaluation. The second layer is a safety net
in the CI/CD pipeline with independent thresholds.

### Workflow 3: deploy.yml --- Deploy to AWS

**File:** `.github/workflows/deploy.yml`

**Triggers:**
- Manual dispatch (`workflow_dispatch`) with environment selection (staging/production)
- Automatically after `train.yml` completes successfully on main

```yaml
name: Deploy to AWS

on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Deployment environment"
        required: true
        default: "production"
        type: choice
        options:
          - staging
          - production
  workflow_run:
    workflows: ["Train & Evaluate Model"]
    types: [completed]
    branches: [main]
```

**The workflow_run trigger** is the key connection between training and deployment. When
the "Train & Evaluate Model" workflow completes (and the condition checks it was successful),
the deployment workflow automatically starts.

**Deployment steps explained:**

```yaml
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' || github.event_name == 'workflow_dispatch' }}
    steps:
      # 1. Configure AWS credentials from GitHub Secrets
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      # 2. Login to ECR
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      # 3. Retrain model for packaging (CI environment has no cached model)
      - name: Install & train model for packaging
        run: |
          pip install -r requirements.txt
          pip install -e .
          python -m src.data.ingest
          python -m src.data.preprocess
          python -m src.models.train

      # 4. Build Docker image with the freshly trained model
      - name: Build Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest

      # 5. Push image to ECR (two tags: sha-based and latest)
      - name: Push Docker image to ECR
        run: |
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      # 6. Update Lambda to use the new image
      - name: Update Lambda function
        run: |
          aws lambda update-function-code \
            --function-name mlops-fraud-prediction \
            --image-uri $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            --region $AWS_REGION

      # 7. Wait for Lambda to finish updating
      - name: Wait for Lambda update
        run: |
          aws lambda wait function-updated \
            --function-name mlops-fraud-prediction \
            --region $AWS_REGION

      # 8. Smoke test the deployed API
      - name: Smoke test
        run: |
          API_URL=$(aws apigatewayv2 get-apis \
            --query "Items[?Name=='fraud-detection-api'].ApiEndpoint" \
            --output text --region $AWS_REGION)
          if [ -n "$API_URL" ]; then
            STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$API_URL/health")
            echo "Health check returned: $STATUS"
            [ "$STATUS" = "200" ] || exit 1
          fi
```

**Dual tagging strategy:** Every image gets two tags:
- `${{ github.sha }}` --- Immutable, tied to the exact commit. Used for Lambda update.
- `latest` --- Mutable, always points to the most recent build. Useful for local testing.

This means you can always trace a deployed Lambda function back to the exact Git commit
that produced it.

---

## Testing Strategy

Our project uses three tiers of tests, each running at a different stage:

```
+-------------------------------------------------------------------+
|                     Testing Pyramid                                |
|                                                                    |
|                        /\                                          |
|                       /  \        Model Quality Tests              |
|                      / MQ \       (tests/model/)                   |
|                     /------\      Run after training (train.yml)   |
|                    /        \                                      |
|                   / Integr.  \    Integration Tests                |
|                  /   Tests    \   (tests/integration/)             |
|                 /--------------\  Run in CI (ci.yml)               |
|                /                \                                  |
|               /   Unit Tests    \ Unit Tests                      |
|              /    (tests/unit/)  \(tests/unit/)                   |
|             /____________________\Run in CI (ci.yml)              |
|                                                                    |
+-------------------------------------------------------------------+
```

### Unit Tests (`tests/unit/`)

**File: `tests/unit/test_preprocess.py`**
Tests preprocessing logic in isolation with synthetic data:
- `test_no_nulls_after_preprocessing` --- Verify no nulls survive preprocessing
- `test_class_column_is_binary` --- Target is only 0 or 1
- `test_scaling_standardizes_amount` --- StandardScaler produces mean near 0, std near 1
- `test_train_test_split_preserves_ratio` --- Stratification preserves class balance
- `test_feature_count_after_drop` --- Dropping Time and Class leaves 29 features

**File: `tests/unit/test_model.py`**
Tests model behavior with a tiny XGBoost model:
- `test_model_outputs_probabilities` --- Probabilities sum to 1.0 per sample
- `test_model_outputs_binary_predictions` --- Predictions are 0 or 1
- `test_model_deterministic` --- Same input produces same output
- `test_model_handles_single_sample` --- Model works with batch size of 1

These tests are fast (< 5 seconds total), have no data dependencies, and catch logical
errors in the pipeline code.

### Integration Tests (`tests/integration/`)

**File: `tests/integration/test_pipeline.py`**
Tests pipeline stages end-to-end:
- `test_ingest_generates_correct_shape` --- Synthetic data has 1000 rows and 31 columns
- `test_ingest_generates_fraud_and_legit` --- Both classes are present
- `test_validate_passes_on_clean_data` --- Clean synthetic data passes validation
- `test_validate_rejects_bad_data` --- Invalid class value (5.0) raises an exception

These tests import actual project code (`src.data.ingest`, `src.data.validate`) and verify
the modules work together correctly.

### Model Quality Tests (`tests/model/`)

**File: `tests/model/test_model_quality.py`**
Tests the trained model against minimum performance thresholds:
- `test_auc_roc_above_threshold` --- AUC-ROC >= 0.90
- `test_recall_above_threshold` --- Recall >= 0.70
- `test_precision_above_threshold` --- Precision >= 0.30
- `test_model_not_predicting_all_same_class` --- Model predicts both fraud and legitimate

These tests read `metrics/eval_metrics.json`, which only exists after the model has been
trained and evaluated. They run in the `train.yml` workflow, not in `ci.yml`.

The `pytest.skip()` call handles the case where metrics do not exist:
```python
def _load_metrics(self) -> dict:
    path = get_project_root() / "metrics" / "eval_metrics.json"
    if not path.exists():
        pytest.skip("eval_metrics.json not found --- run evaluate first")
```

---

## Quality Gates: Preventing Bad Models from Deploying

Quality gates are the ML-specific addition to CI/CD. They are automated checks that verify
a model meets minimum performance requirements before it can be deployed.

```
Quality Gate Architecture:

  Training                  Evaluation              Deployment
  +--------+    model.pkl   +-----------+   metrics  +----------+
  | train  | -------------> | evaluate  | ---------> | deploy-  |
  | .py    |                | .py       |            | gate     |
  +--------+                +-----------+            +----------+
                                |                         |
                                v                         v
                     params.yaml thresholds      train.yml thresholds
                     min_recall: 0.80            auc_roc >= 0.90
                     min_precision: 0.50         recall >= 0.70
                     min_f1: 0.60
                     min_auc_roc: 0.95

                     If ANY fails:               If ANY fails:
                     ValueError raised            Job fails, deploy
                     Pipeline stops               workflow blocked
```

### Three layers of quality enforcement

**Layer 1: evaluate.py (strictest)**
```python
def _check_quality_gates(metrics: dict, params: dict) -> None:
    thresholds = params["thresholds"]
    gates = {
        "recall": ("min_recall", metrics["recall"]),        # >= 0.80
        "precision": ("min_precision", metrics["precision"]),# >= 0.50
        "f1": ("min_f1", metrics["f1"]),                    # >= 0.60
        "auc_roc": ("min_auc_roc", metrics["auc_roc"]),     # >= 0.95
    }
    failures = []
    for metric_name, (threshold_key, actual) in gates.items():
        minimum = thresholds[threshold_key]
        if actual < minimum:
            failures.append(f"{metric_name}: {actual:.4f} < {minimum}")
    if failures:
        raise ValueError("Quality gates FAILED: " + "; ".join(failures))
```

**Layer 2: test_model_quality.py (pytest assertions)**
```python
def test_auc_roc_above_threshold(self):
    metrics = self._load_metrics()
    assert metrics["auc_roc"] >= 0.90

def test_recall_above_threshold(self):
    metrics = self._load_metrics()
    assert metrics["recall"] >= 0.70
```

**Layer 3: train.yml deploy-gate (GitHub Actions)**
```python
assert m['auc_roc'] >= 0.90, f'AUC-ROC too low: {m["auc_roc"]}'
assert m['recall'] >= 0.70, f'Recall too low: {m["recall"]}'
```

Having multiple layers is intentional redundancy. If someone bypasses the evaluation
script (e.g., by modifying `evaluate.py`), the pytest assertions and GitHub Actions checks
still catch bad models.

---

## Branch Strategy for ML Projects

```
main (production)
  |
  |--- develop (integration)
  |      |
  |      |--- feature/new-xgboost-params
  |      |--- feature/add-smote-oversampling
  |      |--- experiment/try-lightgbm
  |      |--- fix/data-leakage-bug
  |
  |--- hotfix/fix-lambda-timeout
```

**Key principles:**

1. **`main` is always deployable.** Every commit to main has passed all quality gates.
   The `train.yml` workflow ensures this.

2. **`develop` is for integration.** Feature branches merge here first. The `ci.yml`
   workflow runs on pushes to develop.

3. **Feature branches for changes.** Each change (new hyperparameters, new features, bug
   fixes) gets its own branch with descriptive naming.

4. **Experiment branches for exploration.** Prefix with `experiment/` to signal that these
   are exploratory and may not merge. The `ci.yml` workflow still runs on PRs to main,
   catching issues early.

5. **Hotfix branches for urgent fixes.** Branch directly from main, fix, and merge back.
   Useful for production issues like Lambda timeout increases.

### PR workflow

```
1. Create feature branch: git checkout -b feature/tune-xgboost
2. Make changes to params.yaml (adjust hyperparameters)
3. Run locally: make pipeline (runs full pipeline)
4. Push and create PR to main
5. ci.yml runs: lint + unit tests + integration tests
6. Code review by team member
7. Merge to main
8. train.yml runs: full pipeline + quality gates
9. If quality gates pass: deploy.yml runs automatically
10. Smoke test verifies the deployment
```

---

## Pre-commit Hooks and Code Quality

Our `.pre-commit-config.yaml` defines hooks that run before every `git commit`:

```yaml
repos:
  # Ruff: fast Python linter + auto-fixer
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]       # Automatically fix simple issues
      - id: ruff-format     # Format code (like Black but faster)

  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace    # Remove trailing spaces
      - id: end-of-file-fixer      # Ensure files end with newline
      - id: check-yaml             # Validate YAML syntax (critical for params.yaml)
      - id: check-json             # Validate JSON syntax (critical for metrics)
      - id: check-added-large-files
        args: [--maxkb=500]        # Block files > 500 KB (catches accidental data commits)
      - id: detect-private-key     # Block private keys from being committed

  # Gitleaks: scan for secrets
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks               # Detect passwords, API keys, tokens
```

**Why each hook matters for ML projects:**

- `check-added-large-files (--maxkb=500)` --- Prevents accidentally committing training
  data, model files, or dataset CSVs to Git. These should go through DVC instead. This
  is the most important ML-specific hook.

- `detect-private-key` and `gitleaks` --- ML projects often involve cloud credentials
  (AWS keys, S3 access). These hooks prevent credentials from being committed. Our project
  has an `AWS_Credential` file that is in `.gitignore` and would be caught by these hooks
  if someone tried to commit it.

- `check-yaml` --- Catches syntax errors in `params.yaml` and `dvc.yaml` before they
  cause confusing pipeline failures.

---

## Interview Questions and Answers

### Q1: How is CI/CD different for ML projects compared to traditional software?

**A:** Three key differences: (1) ML CI/CD must handle data as a first-class artifact
alongside code. A data change can break the model even if no code changed. (2) ML tests
include model quality assertions (AUC-ROC >= 0.95) in addition to traditional unit and
integration tests. These are inherently probabilistic, not deterministic. (3) ML deployment
includes quality gates that compare model performance against minimum thresholds and
potentially against the current production model. Traditional CI/CD only checks if tests
pass; ML CI/CD also checks if the model is "good enough."

### Q2: Walk me through what happens when you push a code change in your project.

**A:** When I push to a feature branch and create a PR to main: (1) `ci.yml` triggers
and runs lint (ruff, black, mypy) followed by unit and integration tests. If any fail,
the PR is blocked. (2) After code review and merge to main, `train.yml` checks if the
changed files match `src/models/**`, `src/data/**`, or `params.yaml`. If yes, it runs
the full pipeline: ingest, validate, preprocess, train, evaluate, model quality tests.
(3) The deploy-gate job checks AUC-ROC >= 0.90 and recall >= 0.70. (4) If gates pass,
`deploy.yml` triggers via `workflow_run`, builds a Docker image, pushes to ECR, updates
the Lambda function, and runs a smoke test against the `/health` endpoint.

### Q3: How do you handle the case where training takes too long for CI/CD?

**A:** Several strategies: (1) Use a smaller sample of data for CI training (our synthetic
generator supports custom `n_samples`). (2) Use DVC's smart caching to skip unchanged
pipeline stages. (3) Separate the CI pipeline into fast checks (lint + unit tests, < 2 min)
and slow checks (training + evaluation, potentially hours). (4) Use `workflow_dispatch` for
manual training triggers instead of triggering on every push. (5) Use GitHub Actions
caching for pip dependencies. (6) Consider training on cloud GPUs (self-hosted runners
with GPU instances) for large models.

### Q4: What is a quality gate and why do you have multiple layers of them?

**A:** A quality gate is an automated check that verifies a model meets minimum performance
requirements. We have three layers: (1) `evaluate.py` checks during the evaluation step
with strict thresholds from `params.yaml`. (2) `test_model_quality.py` runs pytest assertions
as part of the test suite. (3) The `deploy-gate` job in `train.yml` provides a CI-level
check. Multiple layers provide defense in depth --- if someone modifies one layer, the
others still catch bad models. The layered approach also allows different threshold levels:
the evaluation step uses the data scientist's preferred thresholds, while the deployment
gate uses the operations team's minimum acceptable thresholds.

### Q5: How would you implement a rollback strategy for ML deployments?

**A:** Our deployment uses immutable image tags based on the Git commit SHA. To rollback:
(1) Identify the last known good commit SHA. (2) Run
`aws lambda update-function-code --function-name mlops-fraud-prediction --image-uri <ECR_REGISTRY>/<REPO>:<good-sha>`.
(3) Wait for the update and verify with a health check. This works because every deployment
pushes a uniquely tagged image to ECR, and ECR retains the last 10 images (per our lifecycle
policy). For a more sophisticated approach, you could use Lambda aliases with weighted
routing for canary deployments, gradually shifting traffic from the old to new model.

### Q6: How do you prevent data leakage in your CI/CD pipeline?

**A:** Data leakage is prevented at multiple levels: (1) The preprocessing code
(`src/data/preprocess.py`) splits data BEFORE any transformations. The scaler is fit on
training data only and then applied to validation and test sets. (2) Unit tests verify
that train/test splits preserve class ratios (`test_train_test_split_preserves_ratio`).
(3) The evaluate step uses a holdout test set that was never seen during training. (4)
Cross-validation in the training step provides an additional check --- if CV scores and
holdout scores diverge significantly, it suggests leakage. (5) Code review catches
subtle leakage patterns (e.g., using future features in time-series data).

### Q7: How would you add A/B testing to your deployment pipeline?

**A:** I would: (1) Use Lambda aliases (e.g., "production-a" and "production-b") with
weighted routing through API Gateway. (2) Start with 90/10 split (90% old model, 10% new
model). (3) Publish prediction results and business metrics to CloudWatch with a "model_version"
dimension. (4) After sufficient sample size, compare metrics statistically (e.g., chi-squared
test for fraud detection rate differences). (5) If the new model wins, shift to 100%.
If it loses, roll back. (6) This can be automated with a GitHub Actions workflow that
monitors CloudWatch metrics and adjusts the routing weights.

### Q8: What GitHub Actions features do you find most useful for ML CI/CD?

**A:** (1) `workflow_run` trigger --- chains workflows together (train completes, then
deploy starts). (2) `workflow_dispatch` --- allows manual triggering with parameters
(retrain with a documented reason, deploy to a specific environment). (3) `paths` filter ---
only triggers training when ML-relevant files change, saving compute. (4) `actions/upload-artifact`
and `actions/download-artifact` --- passes model files and metrics between jobs without
checking them into Git. (5) Job dependencies with `needs` --- enforces the order: lint ->
test -> train -> quality gate -> deploy. (6) `if` conditions --- `if: github.ref == 'refs/heads/main'`
ensures deployment only happens from main.

---

## Practical Tips

1. **Separate fast and slow checks.** Lint and unit tests run on every push (< 2 min).
   Training runs only when ML-relevant files change (potentially 30+ min). This keeps
   developer feedback fast while ensuring ML quality.

2. **Use `paths` filters aggressively.** There is no reason to retrain when you edit
   documentation, Terraform files, or test files. Our `train.yml` only triggers on
   `src/models/**`, `src/data/**`, and `params.yaml`.

3. **Always include a smoke test after deployment.** Our deploy workflow hits the `/health`
   endpoint after updating Lambda. A more thorough smoke test would also send a sample
   prediction request and verify the response format.

4. **Use immutable image tags.** Tag Docker images with the Git SHA, not just `latest`.
   This enables reliable rollbacks and traceability from a deployed function back to the
   exact code that built it.

5. **Store secrets in GitHub Secrets, never in code.** Our deploy workflow uses
   `${{ secrets.AWS_ACCESS_KEY_ID }}` and `${{ secrets.AWS_SECRET_ACCESS_KEY }}`. The
   pre-commit hooks (`detect-private-key`, `gitleaks`) catch accidental commits.

6. **Upload metrics as artifacts.** The `train` job uploads metrics as a GitHub Actions
   artifact, and the `deploy-gate` job downloads them. This is cleaner than passing data
   through environment variables or files.
