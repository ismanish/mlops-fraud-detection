# Chapter 15: CI/CD Workflows Deep Dive

## Every Workflow, Every Hook, Every Makefile Target Explained Line by Line

This guide tears apart every CI/CD file in our MLOps project. By the end, you will
understand exactly what happens when you push code, open a pull request, trigger a
training run, or deploy to production -- and WHY each line exists.

---

## 1. GitHub Actions Fundamentals

### 1.1 The Vocabulary

Before reading any YAML, you need to know these five terms:

| Term       | What it is                                                                 |
|------------|---------------------------------------------------------------------------|
| **Workflow** | A YAML file in `.github/workflows/`. One file = one automated process.  |
| **Job**      | A named block inside a workflow that runs on a single runner (VM).       |
| **Step**     | A single command or action inside a job. Steps run sequentially.         |
| **Action**   | A reusable unit of code published on the GitHub Marketplace.             |
| **Runner**   | The virtual machine (or physical machine) that executes the job.         |

Think of it as a hierarchy: Workflow > Jobs > Steps > Actions.

A workflow is like a factory floor plan. Jobs are workstations on that floor. Steps are
the sequence of tasks performed at each workstation. Actions are specialized power tools
that someone else built and you plug in.

### 1.2 Trigger Types

```yaml
on:
  push:                    # Fires when commits are pushed
    branches: [main, develop]
  pull_request:            # Fires when a PR is opened/updated
    branches: [main]
  workflow_dispatch:       # Manual trigger via GitHub UI or API
    inputs:
      reason:
        description: "Why?"
        required: true
```

**`on: push`** -- Fires every time someone pushes commits to a matching branch. This is
your "continuous" in continuous integration. Every push gets validated.

**`on: pull_request`** -- Fires when a PR is opened, synchronized (new commits pushed),
or reopened against a matching branch. This is your quality gate -- code must pass CI
before it can merge.

**`on: workflow_dispatch`** -- Adds a "Run workflow" button in the GitHub Actions UI.
Essential for ML because you often need to manually trigger training or deployment. You
can define `inputs` that appear as form fields in the UI.

**Branch filtering with `branches:`** -- Without this filter, a push to ANY branch fires
the workflow. `branches: [main, develop]` means "only fire on pushes to main or develop."
This prevents wasted CI minutes on feature branches that are not ready.

**Path filtering with `paths:`** -- Our train workflow uses this:

```yaml
paths:
  - "src/models/**"
  - "src/data/**"
  - "params.yaml"
```

This means: "only trigger if the push changes files under `src/models/`, `src/data/`, or
the `params.yaml` config." If someone edits the README or a guide file, the training
workflow does NOT run. This saves compute and avoids unnecessary retraining.

### 1.3 Runners

```yaml
runs-on: ubuntu-latest
```

A **runner** is the virtual machine that executes your job. GitHub provides hosted runners
with pre-installed tools (Git, Python, Docker, AWS CLI, etc.). `ubuntu-latest` gives you
a fresh Ubuntu VM with ~7 GB RAM and 2 CPUs.

**GitHub-hosted runners** are free for public repos and have limited free minutes for
private repos. They are ephemeral -- destroyed after the job finishes. Nothing persists
between runs.

**Self-hosted runners** are machines YOU control. Companies use these for:
- GPU access (training large models)
- Access to internal networks
- Custom hardware requirements
- Cost savings at scale (1000+ CI minutes/day)

### 1.4 Job Dependencies

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    # ...
  test:
    runs-on: ubuntu-latest
    needs: lint          # <-- test waits for lint to finish
```

`needs: lint` creates a **dependency**. The `test` job will NOT start until `lint`
succeeds. If `lint` fails, `test` is skipped entirely.

Without `needs`, jobs run **in parallel** by default. GitHub Actions builds a **DAG**
(Directed Acyclic Graph) from your `needs` declarations and runs independent jobs
concurrently.

This is powerful: you could have lint, security scan, and type checking all running in
parallel, with the test job waiting for all three to pass.

### 1.5 Conditional Execution

```yaml
if: github.ref == 'refs/heads/main'
```

The `if:` keyword controls whether a job or step runs. The expression uses **GitHub
context** -- variables that GitHub injects into every workflow run. Common ones:

| Variable               | Example Value                          | Use Case                     |
|------------------------|----------------------------------------|------------------------------|
| `github.ref`           | `refs/heads/main`                      | Branch check                 |
| `github.sha`           | `a1b2c3d4e5f6...`                      | Tagging Docker images        |
| `github.event_name`    | `push`, `pull_request`, `workflow_dispatch` | Trigger type check       |
| `github.actor`         | `manishsingh`                          | Who triggered it             |
| `secrets.AWS_ACCESS_KEY_ID` | `AKIA...` (masked in logs)        | Credentials                  |

### 1.6 GitHub Secrets

```yaml
aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
```

Secrets are encrypted values stored in your repository settings. They are:
- **Encrypted at rest** using libsodium sealed boxes
- **Masked in logs** -- if the value appears in output, GitHub replaces it with `***`
- **Not available to forks** -- a fork's PR cannot read your secrets (security)
- **Scoped** -- repository secrets, environment secrets, or organization secrets

You set them in: Repository > Settings > Secrets and variables > Actions.

### 1.7 Environment Variables

```yaml
env:                          # Workflow-level: available to ALL jobs
  AWS_REGION: us-east-1

jobs:
  deploy:
    env:                      # Job-level: available to all steps in this job
      ECR_REPOSITORY: mlops-fraud-detection
    steps:
      - name: Build
        env:                  # Step-level: available only in this step
          IMAGE_TAG: ${{ github.sha }}
```

Environment variables cascade: workflow > job > step. A step-level variable overrides a
job-level one with the same name. This is how you keep configuration DRY while allowing
step-specific overrides.

### 1.8 Real Example: CI Parallelism

Imagine a large ML project with 500 tests. Without parallelism, CI takes 20 minutes.
With a matrix strategy:

```yaml
strategy:
  matrix:
    test-group: [unit, integration, model, performance, data-quality]
```

Each group runs on its own runner in parallel. 500 tests across 5 runners finish in
4 minutes instead of 20. Our project does not use matrix (we do not have enough tests to
justify it), but the concept matters for interviews.

---

## 2. CI Workflow (ci.yml) -- Line by Line

Let us walk through every single line of `.github/workflows/ci.yml`:

### The Full File

```yaml
name: CI — Lint & Test

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
        run: |
          pip install ruff black mypy
          pip install -r requirements.txt
      - name: Ruff
        run: ruff check src/ tests/
      - name: Black
        run: black --check src/ tests/
      - name: Mypy
        run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint
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

### Line 1: `name: CI — Lint & Test`

The display name shown in the GitHub Actions UI. Use a clear, descriptive name because
when you have 5+ workflows, you need to quickly identify which one failed.

### Lines 3-7: Trigger Configuration

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

**Two triggers, two use cases:**

1. `push` to `main` or `develop` -- validates code that has already been merged. If CI
   fails here, everyone knows the branch is broken and needs immediate attention.

2. `pull_request` to `main` -- validates code BEFORE it merges. This is the gatekeeper.
   GitHub can be configured (branch protection rules) to block merging if CI fails.

Why both? Because someone might push directly to `develop` (bypassing PR), and you still
want validation. Defense in depth.

### Lines 9-11: The Lint Job Header

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
```

`lint` is the job ID (used in `needs:` references). `runs-on: ubuntu-latest` requests a
fresh Ubuntu VM. At the time of writing, `ubuntu-latest` resolves to Ubuntu 22.04 with
pre-installed Docker, Node.js, Python, and 100+ other tools.

### Line 13: `actions/checkout@v4`

```yaml
- uses: actions/checkout@v4
```

This is the MOST used action in all of GitHub. It clones your repository into the runner.

Why is it needed? The runner starts as a blank VM. It has NO copy of your code. Without
`checkout`, every subsequent step would fail because there are no files to lint, test,
or build.

The `@v4` is a version tag. Pinning to a major version (`v4`) means you get bug fixes
and minor improvements automatically, but not breaking changes. For maximum security in
production, you would pin to a specific commit SHA.

### Lines 14-16: `actions/setup-python@v5`

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"
```

The runner has Python pre-installed, but it might not be 3.12. This action:
1. Downloads Python 3.12 if not cached
2. Adds it to `PATH` so `python` and `pip` resolve to 3.12
3. Sets up pip caching (faster subsequent runs)

Why specify the version? Because ML code often depends on specific Python features. A
model trained on 3.12 might use syntax or behavior that does not exist in 3.10. Pinning
the version ensures reproducibility.

### Lines 17-20: Installing Dependencies

```yaml
- name: Install dependencies
  run: |
    pip install ruff black mypy
    pip install -r requirements.txt
```

The `|` (pipe) in YAML means "literal block scalar" -- everything indented below is a
multi-line string. Each line runs as a separate shell command.

**Why install linters FIRST, then requirements?** Order matters for `mypy`. The type
checker needs to see the type stubs for third-party packages. If you install mypy after
`requirements.txt`, it already has access to those stubs. But here we install mypy first,
then requirements, which also works because mypy runs AFTER both install commands. The
real reason for this order is organizational: linting tools are conceptually separate from
project dependencies.

### Lines 21-22: Ruff Linter

```yaml
- name: Ruff
  run: ruff check src/ tests/
```

**What is Ruff?** Ruff is a Python linter written in Rust. It is 10-100x faster than
flake8, pylint, or isort. It can check thousands of files in under a second.

`ruff check src/ tests/` runs Ruff against all Python files in `src/` and `tests/`.
The rules it checks are configured in `pyproject.toml` or `ruff.toml` (not shown here,
but commonly includes):

- **E** -- pycodestyle errors (indentation, whitespace)
- **F** -- pyflakes errors (unused imports, undefined names)
- **I** -- isort rules (import sorting)
- **N** -- pep8-naming (variable/function naming conventions)
- **W** -- pycodestyle warnings
- **UP** -- pyupgrade (modernize Python syntax)

If Ruff finds ANY violation, the step fails with a non-zero exit code, and the entire
lint job fails.

**Interview insight:** Ruff has replaced flake8 + isort + pyupgrade as the standard
Python linter at most modern companies. Know why: it is fast, configurable, and combines
multiple tools into one.

### Lines 23-24: Black Formatter

```yaml
- name: Black
  run: black --check src/ tests/
```

**What is Black?** Black is an opinionated Python code formatter. "Opinionated" means it
makes most formatting decisions for you -- there are very few configuration options.

**`--check` mode** does NOT modify files. It only checks whether the files are already
formatted according to Black's style. If any file would be changed, it exits with code 1
(failure).

This is crucial in CI. You do not want CI reformatting code (that would create new
commits). You want CI to TELL you "your code is not formatted" so you can run
`black src/ tests/` locally and commit the fix.

**Black vs format mode:** Locally, you run `black src/ tests/` (without `--check`) to
actually format files. In CI, you run `black --check` to verify.

### Lines 25-26: Mypy Type Checker

```yaml
- name: Mypy
  run: mypy src/ --ignore-missing-imports
```

**What is Mypy?** Mypy is a static type checker for Python. It reads your type hints
(`def predict(features: dict[str, float]) -> float:`) and verifies they are consistent
throughout the codebase.

**`--ignore-missing-imports`** tells Mypy: "if you cannot find type stubs for a
third-party library, do not report an error." Without this flag, Mypy would error on
every `import pandas` or `import sklearn` because those libraries may not ship with
complete type annotations.

**Why is type checking in CI?** Because Python is dynamically typed, a function might
accept the wrong type and not fail until runtime -- possibly in production. Mypy catches
these errors at build time.

Example of a bug Mypy would catch:

```python
def calculate_risk(score: float) -> str:
    return score > 0.5   # Bug! Returns bool, not str
```

Without Mypy, this code runs fine until something tries to call `.upper()` on the return
value and crashes in production.

### Lines 28-44: The Test Job

```yaml
test:
  runs-on: ubuntu-latest
  needs: lint
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

**`needs: lint`** -- The test job waits for lint to pass. Why? Two reasons:

1. **Fast feedback**: Lint takes 10 seconds, tests take 2 minutes. If lint fails, you
   know within 10 seconds instead of waiting 2 minutes for tests to also fail.
2. **No wasted compute**: If code does not even pass formatting checks, why waste a
   runner running 100 tests?

**`pip install -e .`** -- The `-e` flag stands for "editable install." It installs your
package in development mode by creating a symlink from the site-packages directory to
your source code. This means:

- `import src.models.train` works correctly
- Any code changes are immediately reflected (no need to reinstall)
- The package is importable by pytest

Without `-e .`, pytest might fail with `ModuleNotFoundError: No module named 'src'`
because Python would not know where to find your project's modules.

**`pytest tests/unit/ -m unit -v --tb=short`**

- `tests/unit/` -- Only look for tests in this directory
- `-m unit` -- Only run tests marked with `@pytest.mark.unit`
- `-v` -- Verbose output (show each test name and result)
- `--tb=short` -- Short traceback on failure (just the assertion, not the full stack)

**Why separate unit and integration tests?** Unit tests are fast (mock all external
dependencies) and test individual functions. Integration tests are slower (may use real
databases, APIs, or files) and test how components work together.

Running them separately gives clearer feedback: "unit tests pass but integration tests
fail" tells you the logic is correct but something is wrong with how components connect.

### Real Example: A Lint Error That Would Have Caused a Production Bug

A developer writes:

```python
def preprocess(df):
    df_clean = df.dropna()
    return df   # Bug! Should return df_clean
```

Ruff would not catch this (it is semantically valid Python). But if the developer had
type hints and wrote `-> pd.DataFrame`, Mypy MIGHT flag it depending on context. The
real safety net here is the **test job** -- a unit test that verifies preprocessing
removes null values would catch this. But the lint job catches 80% of issues before
tests even run, saving time.

---

## 3. Train Workflow (train.yml) -- Line by Line

### The Full File

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

env:
  AWS_REGION: us-east-1

jobs:
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

  deploy-gate:
    runs-on: ubuntu-latest
    needs: train
    if: github.ref == 'refs/heads/main'
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
          assert m['recall'] >= 0.02, f'Recall too low: {m[\"recall\"]}'
          print('Quality gates passed — ready for deployment')
          "
```

### Lines 1-16: Name and Triggers

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

**Two triggers for two scenarios:**

1. **`workflow_dispatch`** with an input field -- A data scientist clicks "Run workflow"
   in the GitHub UI, fills in why they are retraining ("Monthly scheduled retrain" or
   "Feature engineering update" or "Data drift detected"), and clicks Go. The `reason`
   input becomes `${{ github.event.inputs.reason }}` in the workflow.

2. **`push` with `paths:`** -- Automatic retraining. If someone pushes changes to model
   code (`src/models/**`), data pipeline code (`src/data/**`), or hyperparameters
   (`params.yaml`), the training pipeline runs automatically. But if someone only changes
   `src/serving/app.py` or `README.md`, the workflow does NOT trigger.

The `**` glob pattern means "match any file at any depth." So `src/models/**` matches
`src/models/train.py`, `src/models/evaluate.py`, and `src/models/utils/feature.py`.

### Lines 18-19: Workflow-Level Environment Variable

```yaml
env:
  AWS_REGION: us-east-1
```

Available to ALL jobs and steps. Setting it at the workflow level avoids repeating
`AWS_REGION: us-east-1` in every step that needs it.

### Lines 22-24: Permissions

```yaml
permissions:
  contents: write
```

**Why does the train workflow need write permission?** By default, the `GITHUB_TOKEN`
(automatically provided to every workflow) has read-only access to repository contents.
`contents: write` allows the workflow to push commits, create releases, or write to the
repository.

In this project, it may be needed if the pipeline writes evaluation results back to the
repository, or if DVC needs to update `.dvc` files. This follows the principle of least
privilege -- only request the permissions you actually need.

### Lines 26-41: The ML Pipeline Steps

```yaml
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
```

This is the **full ML pipeline executing in sequence**. Each step runs one module:

1. **Ingest** -- Download or load raw data
2. **Validate** -- Check data quality (schema, distributions, null counts)
3. **Preprocess** -- Clean, transform, feature engineer
4. **Train** -- Fit the model on preprocessed data
5. **Evaluate** -- Calculate metrics on the test set

The `python -m src.data.ingest` syntax runs the module as a script. This means Python
looks for `src/data/ingest.py` and executes its `if __name__ == "__main__"` block.

**Why separate steps instead of one script?** Each step has its own status in the GitHub
UI. If data validation fails, you see a red X on "Run data validation" specifically --
not a generic "pipeline failed" message. This makes debugging 10x faster.

Also, if training fails, the ingest and validation steps are marked green, so you know
the data is fine and the problem is in the model code.

### Lines 42-43: Model Quality Tests

```yaml
- name: Run model quality tests
  run: pytest tests/model/ -m model -v --tb=short
```

This runs after evaluation and tests things like:
- Is AUC-ROC above minimum threshold?
- Is the model file not empty?
- Can the model make predictions on sample data?
- Is inference time under 100ms?

If these tests fail, the ENTIRE workflow fails, and no artifacts are uploaded. This is
the first quality gate.

### Lines 44-55: Uploading Artifacts

```yaml
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

**What are GitHub Actions artifacts?** Artifacts are files that persist after a workflow
run completes. They are stored for 90 days by default.

**Why do we need them?** The runner (VM) is DESTROYED after the job finishes. Without
uploading artifacts, the trained model and metrics would be lost forever. Artifacts let
us:

1. Download the model from the GitHub UI for inspection
2. Pass data between jobs (the `deploy-gate` job needs the metrics)
3. Keep an audit trail of every model ever trained

`name: model-metrics` is the artifact name (used for downloading later).
`path: metrics/` uploads the entire `metrics/` directory.

### Lines 57-80: The Deploy Gate Job

```yaml
deploy-gate:
  runs-on: ubuntu-latest
  needs: train
  if: github.ref == 'refs/heads/main'
```

**Three critical properties:**

1. **`needs: train`** -- Only runs if the train job succeeded. If training fails or
   quality tests fail, the deploy gate is never reached.

2. **`if: github.ref == 'refs/heads/main'`** -- Only runs on the main branch. If someone
   triggers training from a feature branch (via `workflow_dispatch`), the model trains but
   the deploy gate is skipped. This prevents experimental models from being approved for
   production.

3. **Separate runner** -- This runs on a DIFFERENT VM from the train job. It has no
   access to files from the train job unless they were uploaded as artifacts.

### Lines 62-66: Downloading Artifacts

```yaml
- name: Download metrics
  uses: actions/download-artifact@v4
  with:
    name: model-metrics
    path: metrics/
```

This downloads the `model-metrics` artifact (uploaded by the train job) into the
`metrics/` directory on THIS runner. Now the deploy-gate job can read the evaluation
metrics.

### Lines 67-80: The Quality Gate

```yaml
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
    assert m['recall'] >= 0.02, f'Recall too low: {m[\"recall\"]}'
    print('Quality gates passed — ready for deployment')
    "
```

This is an inline Python script (using `python3 -c`) that acts as the final quality gate.

**Line-by-line breakdown:**

1. Read `metrics/eval_metrics.json` -- the evaluation results from the train job
2. Print all three metrics (AUC-ROC, F1, Recall) for visibility in the logs
3. **Assert AUC-ROC >= 0.90** -- If the model's area under the ROC curve is below 90%,
   the assertion fails, the step exits with a non-zero code, and the entire deploy-gate
   job fails. This prevents a poor model from being deployed.
4. **Assert Recall >= 0.02** -- For fraud detection, recall is critical. Missing a
   fraudulent transaction costs money. Even 2% recall is a low bar (you would normally
   set this much higher), but it ensures the model is at least detecting SOME fraud.
5. Print success message if both gates pass.

**Why use `assert` instead of `if/exit(1)`?** `assert` is concise and produces a clear
error message that includes the actual value. The workflow logs will show:
`AssertionError: AUC-ROC too low: 0.8543`.

**How this prevents bad models from reaching production:** Even if someone accidentally
breaks feature engineering (causing AUC-ROC to drop from 0.95 to 0.80), the deploy gate
catches it. The deploy workflow (next section) is manual, but you would typically only
trigger it after seeing the deploy-gate pass.

### Real Example: Catching a Regression

A data scientist changes the feature engineering code to add a new feature. They do not
realize the new feature has a data leak (it uses future information). The model's
training AUC-ROC jumps to 0.99 (suspiciously high), but the holdout test set AUC-ROC
drops to 0.85. The deploy gate catches this:

```
AssertionError: AUC-ROC too low: 0.8500
```

The team investigates, finds the data leak, and fixes it before a bad model reaches
production.

---

## 4. Deploy Workflow (deploy.yml) -- Line by Line

### The Full File

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

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: mlops-fraud-detection

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install & train model for packaging
        run: |
          pip install -r requirements.txt
          pip install -e .
          python -m src.data.ingest
          python -m src.data.preprocess
          python -m src.models.train

      - name: Build Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Push Docker image to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Update Lambda function
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          aws lambda update-function-code \
            --function-name mlops-fraud-prediction \
            --image-uri $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            --region $AWS_REGION

      - name: Wait for Lambda update
        run: |
          aws lambda wait function-updated \
            --function-name mlops-fraud-prediction \
            --region $AWS_REGION

      - name: Smoke test
        run: |
          API_URL=$(aws apigatewayv2 get-apis \
            --query "Items[?Name=='fraud-detection-api'].ApiEndpoint" \
            --output text --region $AWS_REGION)
          if [ -n "$API_URL" ]; then
            STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$API_URL/health")
            echo "Health check returned: $STATUS"
            [ "$STATUS" = "200" ] || exit 1
          else
            echo "API Gateway not found — skipping smoke test"
          fi
```

### Lines 3-13: Manual Trigger with Choice Input

```yaml
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
```

This workflow is **manual only** (`workflow_dispatch` with no `push` or `pull_request`).
Deployments should NEVER be automatic in production ML systems. A human must decide
"yes, this model is ready."

The `type: choice` input creates a dropdown in the GitHub UI with two options: `staging`
and `production`. This is safer than a free-text field because the operator cannot
accidentally type "prodduction" or "prod" and have the deployment go to the wrong place.

### Lines 21-22: Double Safety Check

```yaml
if: github.event_name == 'workflow_dispatch'
```

This is a belt-and-suspenders check. Even though the workflow only has `workflow_dispatch`
as a trigger, the `if` condition ensures the job ONLY runs when manually dispatched.
This guards against future mistakes -- if someone adds a `push` trigger later, the deploy
job still will not auto-run.

### Lines 27-32: AWS Credential Configuration

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: ${{ env.AWS_REGION }}
```

**How AWS auth works in GitHub Actions:**

1. You store your AWS access key and secret in GitHub Secrets
2. This action reads them and configures the AWS CLI and SDKs on the runner
3. It sets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` as environment
   variables
4. All subsequent steps can use `aws` CLI commands and they will authenticate automatically

**Security note:** In production, you would use OIDC (OpenID Connect) instead of
long-lived access keys. OIDC gives the runner a temporary credential that expires in
1 hour and cannot be stolen. The action supports this via `role-to-assume` parameter.

### Lines 34-36: ECR Login

```yaml
- name: Login to Amazon ECR
  id: login-ecr
  uses: aws-actions/amazon-ecr-login@v2
```

**What is ECR?** Amazon Elastic Container Registry -- a Docker image repository hosted by
AWS.

This action runs `docker login` with temporary ECR credentials. The `id: login-ecr`
assigns an identifier so later steps can reference outputs from this step, specifically:

```
${{ steps.login-ecr.outputs.registry }}
```

This resolves to something like `011015903780.dkr.ecr.us-east-1.amazonaws.com` -- the
ECR registry URL needed for Docker commands.

### Lines 43-48: Train Model for Packaging

```yaml
- name: Install & train model for packaging
  run: |
    pip install -r requirements.txt
    pip install -e .
    python -m src.data.ingest
    python -m src.data.preprocess
    python -m src.models.train
```

This runs the ML pipeline (ingest, preprocess, train) to generate a fresh model file.
The resulting `models/model.pkl` will be baked into the Docker image in the next step.

**Why train during deploy instead of using the artifact from the train workflow?** This is
a design choice. Some teams download the model artifact from the train workflow and
package it. This approach retrains to ensure the model in the Docker image was built from
the exact code being deployed.

Note: data validation is skipped here (`src.data.validate` is not called). The assumption
is that validation already passed in the train workflow.

### Lines 50-57: Building the Docker Image

```yaml
- name: Build Docker image
  env:
    ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
    IMAGE_TAG: ${{ github.sha }}
  run: |
    docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
    docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
      $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

**`IMAGE_TAG: ${{ github.sha }}`** -- Tags the Docker image with the Git commit SHA
(e.g., `a1b2c3d4`). This creates a unique, immutable tag for every deployment. You can
always trace a running container back to the exact commit that built it.

**`docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .`**

Builds the Docker image using the `Dockerfile` in the current directory (`.`). The full
tag looks like:
```
011015903780.dkr.ecr.us-east-1.amazonaws.com/mlops-fraud-detection:a1b2c3d4
```

**`docker tag ... :latest`** -- Also tags the same image as `latest`. This is a
convenience pointer -- `latest` always points to the most recent deployment.

### Lines 59-65: Pushing to ECR

```yaml
- name: Push Docker image to ECR
  env:
    ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
    IMAGE_TAG: ${{ github.sha }}
  run: |
    docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

Two pushes: the SHA-tagged image and the `latest` tag. Both point to the same image
layers, so the second push is nearly instant (Docker detects the layers already exist in
ECR).

### Lines 67-75: Updating Lambda

```yaml
- name: Update Lambda function
  env:
    ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
    IMAGE_TAG: ${{ github.sha }}
  run: |
    aws lambda update-function-code \
      --function-name mlops-fraud-prediction \
      --image-uri $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
      --region $AWS_REGION
```

This tells AWS Lambda: "use this new Docker image for the function." Lambda pulls the
image from ECR and prepares a new execution environment.

**Why use the SHA tag, not `latest`?** Immutability. The SHA tag is permanent. If you
point Lambda to `latest` and someone pushes a broken image, Lambda would pick it up.
With a SHA tag, the function is pinned to a specific, tested image.

### Lines 77-81: Waiting for Update

```yaml
- name: Wait for Lambda update
  run: |
    aws lambda wait function-updated \
      --function-name mlops-fraud-prediction \
      --region $AWS_REGION
```

`aws lambda wait function-updated` polls Lambda every 5 seconds until the function's
state changes from `InProgress` to `Active`. This is crucial because the next step (smoke
test) needs the new code to be live.

Without this wait, the smoke test might hit the OLD version of the function and pass even
though the new version is broken.

### Lines 83-92: Smoke Test

```yaml
- name: Smoke test
  run: |
    API_URL=$(aws apigatewayv2 get-apis \
      --query "Items[?Name=='fraud-detection-api'].ApiEndpoint" \
      --output text --region $AWS_REGION)
    if [ -n "$API_URL" ]; then
      STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$API_URL/health")
      echo "Health check returned: $STATUS"
      [ "$STATUS" = "200" ] || exit 1
    else
      echo "API Gateway not found — skipping smoke test"
    fi
```

**Line-by-line:**

1. **`aws apigatewayv2 get-apis --query "Items[?Name=='fraud-detection-api'].ApiEndpoint"`**
   -- Uses JMESPath query to find the API Gateway endpoint URL by name. Returns something
   like `https://abc123.execute-api.us-east-1.amazonaws.com`.

2. **`if [ -n "$API_URL" ]`** -- Check if the URL is non-empty. If API Gateway does not
   exist yet, skip the test gracefully instead of failing.

3. **`curl -s -o /dev/null -w '%{http_code}' "$API_URL/health"`** -- Hit the `/health`
   endpoint silently (`-s`), discard the response body (`-o /dev/null`), and only print
   the HTTP status code (`-w '%{http_code}'`).

4. **`[ "$STATUS" = "200" ] || exit 1`** -- If the status is not 200, the step fails.
   This catches scenarios where the deployment succeeded but the application is broken
   (model file missing, dependency error, etc.).

### Real Example: Blue-Green Deployment for ML Models

In production, you would want zero-downtime deployments. The strategy:

1. Deploy new model to a separate Lambda function (the "green" environment)
2. Run smoke tests and shadow traffic against green
3. Switch API Gateway to point to green
4. Keep the old function (blue) running for 24 hours as a rollback target
5. After 24 hours with no issues, delete blue

Our workflow does a simpler in-place update, but the concepts are the same. The smoke test
is the minimum viable health check.

---

## 5. Pre-commit Hooks (.pre-commit-config.yaml) -- Every Hook Explained

### What Are Pre-commit Hooks?

Pre-commit hooks are scripts that run BEFORE `git commit` executes. If any hook fails,
the commit is REJECTED. This catches issues at the earliest possible moment -- on the
developer's machine, before code even reaches GitHub.

The `pre-commit` framework (https://pre-commit.com) manages these hooks. It downloads
and caches the hook implementations so you do not need to install them globally.

### The Full File

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: detect-private-key

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks
```

### Hook 1: `ruff` with `--fix`

```yaml
- id: ruff
  args: [--fix]
```

Runs Ruff linter on all staged Python files. The `--fix` flag means it AUTOMATICALLY
fixes issues it can (like removing unused imports, sorting imports). The fixed files are
saved but NOT automatically staged -- you need to `git add` them again.

This is different from CI (where Ruff runs without `--fix`). Locally, you want auto-fix
for convenience. In CI, you want strict checking (no auto-fixing because the CI runner
should not modify code).

### Hook 2: `ruff-format`

```yaml
- id: ruff-format
```

Ruff's formatter (an alternative to Black). It formats code according to a consistent
style. Like `ruff --fix`, it modifies files in-place. If any file was reformatted, the
hook "fails" to alert you that the file changed -- you re-stage and re-commit.

### Hook 3: `trailing-whitespace`

```yaml
- id: trailing-whitespace
```

Removes whitespace at the end of lines. Trailing whitespace is invisible but:
- Creates noisy diffs (the whitespace shows up as a change)
- Can cause issues in YAML files (where trailing spaces can be significant)
- Is considered sloppy by code reviewers

### Hook 4: `end-of-file-fixer`

```yaml
- id: end-of-file-fixer
```

Ensures every file ends with exactly one newline character. This is a POSIX standard --
many Unix tools expect a trailing newline. Without it, `cat file1 file2` might merge the
last line of file1 with the first line of file2.

### Hook 5: `check-yaml`

```yaml
- id: check-yaml
```

Validates that all YAML files (`.yml`, `.yaml`) are syntactically valid. This catches
common YAML mistakes like:
- Incorrect indentation (YAML is whitespace-sensitive)
- Mixing tabs and spaces
- Missing colons in key-value pairs
- Invalid special characters

This is essential for an MLOps project where `params.yaml`, `dvc.yaml`, and workflow
files are all YAML.

### Hook 6: `check-json`

```yaml
- id: check-json
```

Validates JSON files are syntactically valid. Catches trailing commas (invalid in JSON
but valid in JavaScript), missing quotes on keys, and unmatched brackets.

### Hook 7: `check-added-large-files`

```yaml
- id: check-added-large-files
  args: [--maxkb=500]
```

**This is CRITICAL for ML projects.** It blocks any file larger than 500 KB from being
committed. Why?

- Data files (CSV, Parquet) can be gigabytes. Git is terrible at handling large files.
- Model files (`.pkl`, `.h5`) can be hundreds of megabytes.
- Once a large file is in Git history, it is there FOREVER (even if you delete it later).

This hook forces developers to use DVC or Git LFS for large files instead of committing
them directly. The 500 KB limit is generous enough for code and small configs but blocks
data and model files.

### Hook 8: `detect-private-key`

```yaml
- id: detect-private-key
```

Scans staged files for patterns that look like private keys:
- RSA private keys (`-----BEGIN RSA PRIVATE KEY-----`)
- EC private keys
- PGP private keys
- SSH private keys

If you accidentally copy-paste a private key into a config file or notebook, this hook
stops you from committing it.

### Hook 9: `gitleaks`

```yaml
- repo: https://github.com/gitleaks/gitleaks
  rev: v8.18.1
  hooks:
    - id: gitleaks
```

**Gitleaks** is a more advanced secret scanner. While `detect-private-key` only catches
key file formats, gitleaks uses regex patterns and entropy analysis to detect:

- AWS access keys (`AKIA...`)
- AWS secret keys (high-entropy strings near `aws_secret`)
- API tokens (GitHub, Slack, Stripe, etc.)
- Database connection strings with passwords
- Generic high-entropy strings that look like secrets

It uses a rules file with 100+ patterns for different credential types.

### Real Example: Gitleaks Catching an AWS Key

A developer creates a test script and hardcodes their AWS credentials:

```python
# test_s3_connection.py
import boto3
client = boto3.client('s3',
    aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
    aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
)
```

They try to commit:

```
$ git add test_s3_connection.py
$ git commit -m "test s3"

gitleaks...........................................................Failed
- hook id: gitleaks
- exit code: 1

Finding: AKIAIOSFODNN7EXAMPLE
Rule:    aws-access-key-id
File:    test_s3_connection.py
Line:    4
```

The commit is BLOCKED. The key never reaches GitHub. Without this hook, the key would be
in the repository history forever, and AWS credential scanners would detect it within
minutes (yes, attackers actively scan GitHub for leaked keys).

---

## 6. Makefile -- The Command Center

### What is Make?

Make is a build automation tool from 1976. Originally designed for compiling C programs,
it has been adopted by ML teams as a "command center" for running complex multi-step
workflows with simple commands.

**Why use Make for ML projects?**

1. **Discoverability** -- New team members run `make help` or read the Makefile to see all
   available commands.
2. **Standardization** -- Everyone uses the same commands. No "it works on my machine."
3. **Chaining** -- `make pipeline` can run ingest, validate, preprocess, train, evaluate
   in sequence with one command.
4. **Documentation** -- The Makefile IS the documentation for how to run things.

### The Full Makefile

```makefile
.PHONY: setup install lint test train evaluate serve deploy monitor clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pre-commit install
	dvc init
	@echo "Setup complete. Activate venv: source .venv/bin/activate"

install:
	pip install -r requirements.txt

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/

test-unit:
	pytest tests/unit/ -m unit -v

test-integration:
	pytest tests/integration/ -m integration -v

test-model:
	pytest tests/model/ -m model -v

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

ingest:
	python -m src.data.ingest

validate:
	python -m src.data.validate

preprocess:
	python -m src.data.preprocess

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

pipeline: ingest validate preprocess train evaluate
	@echo "Full pipeline complete."

serve-local:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t mlops-fraud-detection:latest .

docker-run:
	docker run -p 8000:8000 mlops-fraud-detection:latest

ecr-push:
	./scripts/deploy.sh push

deploy-lambda:
	./scripts/deploy.sh deploy

deploy: docker-build ecr-push deploy-lambda

monitor-drift:
	python -m src.monitoring.drift_detection

monitor-performance:
	python -m src.monitoring.performance

dvc-push:
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf mlruns/ mlartifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

### Line 1: `.PHONY`

```makefile
.PHONY: setup install lint test train evaluate serve deploy monitor clean
```

**What are phony targets?** Make was designed for file-based builds. When you say
`make train`, Make checks if a FILE called `train` exists and whether it is newer than its
dependencies. If the file exists and is up-to-date, Make says "nothing to do."

`.PHONY` tells Make: "these targets are NOT files. Always run the recipe, even if a file
with that name exists." Without this, if you had a directory called `test/`, running
`make test` would say "`test` is up to date" and do nothing.

### Target: `setup`

```makefile
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pre-commit install
	dvc init
	@echo "Setup complete. Activate venv: source .venv/bin/activate"
```

The **onboarding command**. A new team member clones the repo and runs `make setup`. This:

1. **`python3 -m venv .venv`** -- Creates a virtual environment in `.venv/`. Isolates
   project dependencies from the system Python.
2. **`.venv/bin/pip install --upgrade pip`** -- Upgrades pip to the latest version. Old pip
   versions may not handle modern dependency resolution correctly.
3. **`.venv/bin/pip install -r requirements.txt`** -- Installs all project dependencies.
   Uses the venv's pip specifically (not the system pip).
4. **`.venv/bin/pre-commit install`** -- Installs the pre-commit hooks into `.git/hooks/`.
   After this, every `git commit` triggers the hooks automatically.
5. **`dvc init`** -- Initializes DVC in the repository for data version control.
6. **`@echo`** -- The `@` suppresses Make from printing the command itself. Without `@`,
   you would see both `echo "Setup complete..."` and `Setup complete...`.

### Target: `install`

```makefile
install:
	pip install -r requirements.txt
```

A simpler install that assumes you are already in an activated virtual environment.
Used when you just need to update dependencies without full setup.

### Targets: `lint` and `format`

```makefile
lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/
```

**`lint`** -- READ-ONLY checking. Reports problems but does not modify files. Mirrors what
CI does. Run this before pushing to catch issues locally.

**`format`** -- WRITE mode. Actually fixes formatting issues. `ruff --fix` auto-fixes
linting issues. `black` (without `--check`) reformats files in place. Run this, then
`git add` the changes.

### Test Targets

```makefile
test-unit:
	pytest tests/unit/ -m unit -v

test-integration:
	pytest tests/integration/ -m integration -v

test-model:
	pytest tests/model/ -m model -v

test:
	pytest tests/ -v --cov=src --cov-report=term-missing
```

Four levels of testing:
- **`test-unit`** -- Fast tests (~5 seconds). Run after every code change.
- **`test-integration`** -- Medium tests (~30 seconds). Run before pushing.
- **`test-model`** -- Model quality tests. Run after training.
- **`test`** -- ALL tests with coverage report. Run before major releases.

The `--cov=src --cov-report=term-missing` flags generate a coverage report showing which
lines in `src/` are NOT covered by tests. Example output:

```
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
src/data/ingest.py             45      3    93%   67-69
src/models/train.py            80      0   100%
```

### Pipeline Targets

```makefile
ingest:
	python -m src.data.ingest

validate:
	python -m src.data.validate

preprocess:
	python -m src.data.preprocess

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

pipeline: ingest validate preprocess train evaluate
	@echo "Full pipeline complete."
```

Each ML step has its own target so you can run them individually during development.
**`pipeline`** chains all five by listing them as dependencies. Make runs them left to
right: ingest first, then validate, then preprocess, then train, then evaluate.

If any step fails, Make stops immediately. If `validate` fails (data quality issue), the
pipeline does not proceed to `preprocess` -- preventing wasted compute on bad data.

### Serving and Docker Targets

```makefile
serve-local:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t mlops-fraud-detection:latest .

docker-run:
	docker run -p 8000:8000 mlops-fraud-detection:latest
```

**`serve-local`** -- Starts the FastAPI server locally with hot-reload. Changes to Python
files automatically restart the server. Used during development.

**`docker-build`** and **`docker-run`** -- Build and run the Docker container locally.
Used to test the containerized version before pushing to ECR.

### Deployment Targets

```makefile
ecr-push:
	./scripts/deploy.sh push

deploy-lambda:
	./scripts/deploy.sh deploy

deploy: docker-build ecr-push deploy-lambda
```

**`deploy`** chains three targets: build the Docker image, push to ECR, update Lambda.
This is the full deployment pipeline in one command. But typically you would use the
GitHub Actions deploy workflow instead (for audit trail and access control).

### Monitoring Targets

```makefile
monitor-drift:
	python -m src.monitoring.drift_detection

monitor-performance:
	python -m src.monitoring.performance
```

Run drift detection or performance monitoring manually. In production, these would be
scheduled (cron job or GitHub Actions schedule).

### DVC and MLflow Targets

```makefile
dvc-push:
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --port 5000
```

**`dvc-push`** -- Upload tracked data and model files to remote storage (S3).
**`dvc-pull`** -- Download tracked data and model files from remote storage.
**`mlflow-ui`** -- Start the MLflow experiment tracking UI on port 5000.

### Target: `clean`

```makefile
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf mlruns/ mlartifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

Removes ALL generated files:
- `__pycache__`, `.pyc` -- Python bytecode caches
- `.pytest_cache` -- Pytest result caches
- `.mypy_cache` -- Mypy type checking caches
- `.ruff_cache` -- Ruff linting caches
- `mlruns/`, `mlartifacts/` -- MLflow experiment data

The `2>/dev/null || true` suppresses errors (e.g., if the directories do not exist) and
ensures the command always succeeds (so Make does not stop on a "file not found" error).

### Real Example: Onboarding a New Team Member

Day 1 at a new ML engineering job:

```bash
git clone https://github.com/company/fraud-detection.git
cd fraud-detection
make setup     # 2 minutes: venv, deps, hooks, DVC
source .venv/bin/activate
make test      # Verify everything works
make pipeline  # Run the full ML pipeline
make serve-local  # Start the API locally
```

In 5 minutes, the new engineer has a fully working development environment with all
tools configured. Compare this to a README with 47 manual steps that nobody keeps updated.

---

## Summary: The CI/CD Safety Net

Our three workflows form a layered safety net:

```
Developer Machine          GitHub Actions              AWS
-------------------        ------------------          ----------
pre-commit hooks    --->   CI (lint + test)    --->    (nothing yet)
  ruff                       ruff
  ruff-format                black --check
  trailing-whitespace        mypy
  check-yaml                 pytest unit
  check-large-files          pytest integration
  detect-private-key
  gitleaks

                           Train & Evaluate    --->    (nothing yet)
                             ingest
                             validate
                             preprocess
                             train
                             evaluate
                             model quality tests
                             deploy gate

                           Deploy              --->    ECR push
                             AWS auth                  Lambda update
                             Docker build              Smoke test
                             Docker push
```

Each layer catches different types of issues:
- **Pre-commit**: formatting, secrets, large files (instant feedback)
- **CI**: linting, type errors, test failures (2-minute feedback)
- **Train**: model quality, data issues, metric regressions (10-minute feedback)
- **Deploy**: infrastructure issues, runtime errors (5-minute feedback)

**Interview tip:** When asked about CI/CD for ML, emphasize that ML pipelines have
ADDITIONAL gates beyond traditional software: data validation, model quality testing, and
metric-based deploy gates. A passing test suite does not mean the model is good -- you
need metric thresholds (AUC-ROC, recall) as explicit quality gates.
