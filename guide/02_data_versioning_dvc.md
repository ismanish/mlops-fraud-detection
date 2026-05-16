# Data Versioning with DVC

## Why Data Versioning Matters

In traditional software engineering, Git is enough. You version your code, and you can
reproduce any past state of the application. In ML, code alone is insufficient. The same
code trained on different data produces a different model. If you cannot reproduce the
exact data that was used to train a model, you cannot reproduce the model. Period.

Consider this scenario: your fraud detection model was trained six months ago and has been
working well. Today, a regulator asks you to explain why a specific transaction was flagged.
You need to:

1. Find the exact model version that made the prediction
2. Find the exact training data that produced that model
3. Find the exact preprocessing pipeline that transformed the data
4. Retrain the model to verify it produces the same result

Without data versioning, step 2 is impossible. The training data might have been overwritten,
augmented with new transactions, or simply deleted to save storage. DVC (Data Version Control)
solves this problem.

### The problem with Git for data

Git stores the full content of every file version. A 1 GB CSV with 1000 commits would consume
roughly 1 TB of storage (Git does compress, but the point stands). Git was designed for text
files that change incrementally, not for large binary files or datasets that change wholesale.

```
Git for code:     Small diffs, efficient delta compression
Git for data:     Full copies, repository bloats, clone takes hours

DVC for data:     Only metadata (.dvc files) in Git
                  Actual data in cheap remote storage (S3, GCS, Azure Blob)
```

---

## DVC Architecture: How It Works Under the Hood

DVC operates as a layer on top of Git. It stores lightweight pointer files (`.dvc` files)
in Git and manages the actual data in a separate storage layer.

```
+----------------------------------------------------------+
|                      Git Repository                       |
|                                                           |
|  Code:        src/data/ingest.py                         |
|               src/models/train.py                        |
|               ...                                         |
|                                                           |
|  DVC Metadata: data/raw/creditcard.csv.dvc               |
|                models/model.pkl.dvc                       |
|                dvc.yaml (pipeline definition)             |
|                dvc.lock (exact pipeline state)            |
|                                                           |
|  Params:       params.yaml                               |
|  Metrics:      metrics/eval_metrics.json (cache: false)  |
+----------------------------------------------------------+
          |                                |
          |  git push/pull                |  dvc push/pull
          v                                v
+------------------+        +---------------------------+
|  GitHub           |        |  S3 Remote                |
|  (code + metadata)|        |  s3://mlops-fraud-        |
|                    |        |    detection-011015903780 |
|                    |        |  /data/raw/creditcard.csv |
|                    |        |  /data/processed/*.csv    |
|                    |        |  /models/model.pkl        |
+------------------+        +---------------------------+
```

### .dvc Files

A `.dvc` file is a small YAML file that stores the MD5 hash, size, and path of a tracked
file. When you run `dvc add data/raw/creditcard.csv`, DVC:

1. Computes the MD5 hash of the file
2. Moves the file to the DVC cache (`.dvc/cache/`)
3. Creates a symlink from the original path to the cache
4. Generates `data/raw/creditcard.csv.dvc` containing the hash
5. Adds `data/raw/creditcard.csv` to `.gitignore`

Example `.dvc` file:
```yaml
outs:
  - md5: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
    size: 150828752
    path: creditcard.csv
```

This file is only ~100 bytes. It goes into Git. The 150 MB CSV goes into S3.

### DVC Cache

The DVC cache lives at `.dvc/cache/` in your project root. It uses content-addressable
storage, meaning files are stored by their MD5 hash. If two versions of the data produce
the same hash, only one copy is stored.

```
.dvc/cache/
|-- a1/
|   |-- b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4   # creditcard.csv (version 1)
|-- f7/
|   |-- 89abcdef0123456789abcdef01234567   # creditcard.csv (version 2)
```

This is the same design Git uses for its object storage, applied to large files.

### DVC Remotes

A DVC remote is like a Git remote but for data. Instead of GitHub, it points to object
storage like S3, GCS, or Azure Blob.

```bash
# Configure S3 as our DVC remote
dvc remote add -d myremote s3://mlops-fraud-detection-011015903780

# Push data to S3
dvc push

# Pull data from S3 (new team member or CI/CD)
dvc pull
```

Our project uses S3 as the remote, configured in the bucket
`mlops-fraud-detection-011015903780`. The setup script (`scripts/setup_aws.sh`) creates this
bucket with versioning enabled and public access blocked.

---

## DVC Pipeline Stages: Our dvc.yaml Walkthrough

DVC pipelines define a DAG (Directed Acyclic Graph) of stages, each with commands,
dependencies, parameters, outputs, and metrics. This is where DVC goes beyond simple file
tracking and becomes a full pipeline orchestration tool.

### Our pipeline DAG

```
params.yaml
    |
    v
+----------+     +----------+     +-------------+     +--------+     +----------+
|  ingest  | --> | validate | --> | preprocess  | --> |  train | --> | evaluate |
+----------+     +----------+     +-------------+     +--------+     +----------+
    |                 |                  |                  |              |
    v                 |                  v                  v              v
creditcard.csv        |           X_train.csv         model.pkl     eval_metrics.json
                      |           X_val.csv                         confusion_matrix.json
                      |           X_test.csv                        roc_curve.json
                      |           y_train.csv
                      |           y_val.csv
                      |           y_test.csv
                      |           scaler.pkl
                      |
            (no outputs, just validation)
```

### Stage-by-stage breakdown

**Stage 1: ingest**
```yaml
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps:
      - src/data/ingest.py          # If this code changes, rerun
      - params.yaml                 # If params change, rerun
    outs:
      - data/raw/creditcard.csv:    # Output tracked by DVC
          cache: true               # Store in DVC cache + remote
    params:
      - data.raw_path              # Only rerun if this specific param changes
```

What this does: Downloads or generates the raw credit card fraud dataset (284,807
transactions, 31 columns). The `params:` section is smart --- DVC only reruns this stage
if `data.raw_path` changes in `params.yaml`, not if you change an unrelated parameter like
`model.params.max_depth`.

The `cache: true` setting means the output CSV is stored in DVC's cache and can be pushed
to the S3 remote. This is the default behavior.

**Stage 2: validate**
```yaml
  validate:
    cmd: python -m src.data.validate
    deps:
      - src/data/validate.py
      - data/raw/creditcard.csv     # Depends on ingest output
    params:
      - data
      - features
```

What this does: Runs pandera schema validation on the raw data. Checks column types (all
floats), value ranges (Time >= 0, Amount >= 0), class values (only 0.0 or 1.0), null values,
duplicates, and fraud ratio sanity (< 0.5). Note: this stage has no `outs:` because it
validates in place --- it either passes or raises an exception.

**Stage 3: preprocess**
```yaml
  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - src/data/validate.py        # Also depends on validate code
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
      - data                        # test_size, val_size, random_state
      - features                    # drop_columns, target_column, scaling_method
```

What this does: Drops the `Time` column, splits into train/val/test (70/10/20 with
stratification to preserve fraud ratio), scales the `Amount` column with StandardScaler,
and saves everything. The scaler is serialized as a pickle for use at inference time.

Notice the `deps:` includes `src/data/validate.py` because `preprocess.py` imports and
calls `validate_data()` internally. DVC needs to know about this dependency to correctly
determine when to rerun.

**Stage 4: train**
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
      - model                       # type, params (n_estimators, max_depth, etc.)
      - training                    # experiment_name, cv_folds, early_stopping_rounds
    metrics:
      - metrics/train_metrics.json:
          cache: false              # Always commit to Git (for dvc metrics diff)
```

What this does: Trains an XGBoost classifier with parameters from `params.yaml`. Runs 5-fold
stratified cross-validation. Logs everything to MLflow (parameters, metrics, model artifact).
Saves the model as `model.pkl` and writes training metrics to JSON.

The `cache: false` on metrics is important --- it tells DVC not to cache this file in its
storage but to let Git track it directly. This enables `dvc metrics diff` to compare metrics
across Git commits without needing to pull from the remote.

**Stage 5: evaluate**
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

What this does: Evaluates the model on the holdout test set. Computes precision, recall, F1,
AUC-ROC, and average precision. Generates confusion matrix and ROC curve data. Checks quality
gates (min recall 0.80, min precision 0.50, min F1 0.60, min AUC-ROC 0.95). Raises ValueError
if any gate fails.

The `plots:` section enables `dvc plots show` and `dvc plots diff` to visualize confusion
matrices and ROC curves directly from the command line or in VS Code.

---

## DVC + S3 Remote Storage

### Setting up the remote

```bash
# Initialize DVC in an existing Git repo
dvc init

# Add S3 as the default remote
dvc remote add -d storage s3://mlops-fraud-detection-011015903780

# Optionally set the region
dvc remote modify storage region us-east-1
```

This creates `.dvc/config`:
```ini
[core]
    remote = storage
['remote "storage"']
    url = s3://mlops-fraud-detection-011015903780
    region = us-east-1
```

### How push/pull works

```
Developer A (trains new model):
  1. dvc repro          # Runs pipeline, generates new model.pkl
  2. git add dvc.lock   # Lock file captures exact pipeline state
  3. git commit -m "Retrain with new hyperparams"
  4. dvc push           # Uploads data/models to S3
  5. git push           # Pushes code + DVC metadata to GitHub

Developer B (reproduces the result):
  1. git pull            # Gets code + DVC metadata
  2. dvc pull            # Downloads data/models from S3
  3. dvc repro           # Skips all stages (nothing changed)
  # OR
  3. python -m src.models.evaluate  # Can run directly, data is here
```

### S3 bucket structure

```
s3://mlops-fraud-detection-011015903780/
|-- files/md5/           # DVC cache (content-addressable)
|   |-- a1/
|   |   |-- b2c3d4...   # creditcard.csv hash
|   |-- f7/
|       |-- 89abcd...   # model.pkl hash
|-- data/raw/            # Also used by setup_aws.sh for folder structure
|-- data/processed/
|-- models/
|-- metrics/
|-- drift-reports/
```

DVC uses the `files/md5/` prefix to store cached files. The folder structure under
`data/`, `models/`, etc. is created by `setup_aws.sh` for organizational purposes
but is not used by DVC directly.

---

## DVC Commands Cheat Sheet

### Setup and Configuration
```bash
dvc init                          # Initialize DVC in a Git repo
dvc remote add -d <name> <url>    # Add a default remote
dvc remote modify <name> <key> <val>  # Modify remote settings
dvc remote list                   # List configured remotes
```

### Tracking Data
```bash
dvc add data/raw/creditcard.csv   # Start tracking a file
dvc remove data/raw/creditcard.csv.dvc  # Stop tracking
```

### Pipeline Operations
```bash
dvc repro                         # Reproduce the full pipeline
dvc repro train                   # Reproduce up to and including "train" stage
dvc repro --force                 # Force rerun all stages
dvc repro --dry                   # Show what would run without running
dvc dag                           # Visualize the pipeline DAG (ASCII)
dvc dag --dot | dot -Tpng > dag.png  # Export DAG as image
```

### Data Transfer
```bash
dvc push                          # Push cached files to remote
dvc push data/raw/creditcard.csv  # Push specific file
dvc pull                          # Pull all tracked files from remote
dvc pull data/raw/creditcard.csv  # Pull specific file
dvc fetch                         # Download to cache without checkout
dvc checkout                      # Checkout files from local cache
```

### Metrics and Comparisons
```bash
dvc metrics show                  # Show current metrics
dvc metrics diff                  # Compare metrics between commits
dvc metrics diff HEAD~1           # Compare with previous commit
dvc plots show                    # Generate plots from plot files
dvc plots diff                    # Compare plots across commits
dvc params diff                   # Show parameter changes between commits
```

### Inspection
```bash
dvc status                        # Show which stages are outdated
dvc status --cloud                # Check if remote is in sync
dvc diff                          # Show data changes between commits
dvc version                       # Show DVC version info
```

---

## How dvc repro Works (Smart Caching)

This is one of DVC's most powerful features. When you run `dvc repro`, DVC:

1. Reads `dvc.yaml` to understand the pipeline DAG
2. Reads `dvc.lock` to understand what ran last time (exact hashes of inputs/outputs)
3. Computes current hashes of all dependencies
4. Compares current hashes with `dvc.lock` hashes
5. Only reruns stages whose inputs have changed

```
Scenario: You change model.params.max_depth from 6 to 8 in params.yaml

dvc repro output:
  Stage 'ingest' didn't change, skipping      # data.raw_path unchanged
  Stage 'validate' didn't change, skipping     # data and features unchanged
  Stage 'preprocess' didn't change, skipping   # data and features unchanged
  Running stage 'train'                        # model.params changed!
  Running stage 'evaluate'                     # depends on model.pkl which changed
```

This saves enormous time. If your data ingestion takes 30 minutes and preprocessing takes
15 minutes, but you only changed a hyperparameter, DVC skips 45 minutes of work and only
reruns training and evaluation.

### dvc.lock

The `dvc.lock` file is auto-generated by `dvc repro`. It records the exact state of every
stage's inputs and outputs at the time it was run. This is what enables the smart caching.

```yaml
schema: '2.0'
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps:
    - path: src/data/ingest.py
      hash: md5
      md5: abc123...
    - path: params.yaml
      hash: md5
      md5: def456...
    params:
      params.yaml:
        data.raw_path: data/raw/creditcard.csv
    outs:
    - path: data/raw/creditcard.csv
      hash: md5
      md5: 789ghi...
      size: 150828752
```

The `dvc.lock` file should be committed to Git. It serves as a record of exactly which
inputs produced which outputs for every pipeline run.

---

## Comparison: DVC vs Alternatives

| Feature            | DVC               | Git LFS           | Delta Lake          | LakeFS              |
|--------------------|--------------------|--------------------|---------------------|---------------------|
| **Approach**       | Git extension      | Git extension      | Storage format      | Git-like data layer |
| **Versioning**     | File-level hashing | File-level         | Table-level ACID    | Object-level        |
| **Pipelines**      | Built-in DAG       | None               | None (use Spark)    | None (use external) |
| **Experiment tracking** | Via dvc.yaml + params | None         | None                | None                |
| **Storage backend**| S3, GCS, Azure, local | Git server      | S3, HDFS, ADLS      | S3, GCS, Azure      |
| **Branching**      | Git branches       | Git branches       | No branching        | Native branching    |
| **Merge**          | Via Git            | Via Git            | MERGE INTO          | Branch merge        |
| **Scale**          | Good for files <10GB each | <2GB/file  | Petabyte-scale      | Petabyte-scale      |
| **Learning curve** | Moderate (Git-like)| Low                | High (Spark needed) | Moderate            |
| **Best for**       | ML pipelines       | Large single files | Data warehouses     | Data lakes at scale |
| **Cost**           | Free + storage     | Git hosting limits | Free + compute      | Free + storage      |

### When to use what

- **DVC:** Best for ML projects where you need pipelines + data versioning + experiment
  tracking in a Git-centric workflow. Exactly what we use in this project.

- **Git LFS:** Best for versioning a few large files (game assets, design files) without
  pipelines. Not great for ML because it has no pipeline support and 2 GB file limits on
  most hosted Git services.

- **Delta Lake:** Best for large-scale data warehousing where you need ACID transactions,
  time travel, and schema evolution on massive datasets. Overkill for an ML training pipeline.

- **LakeFS:** Best for data lake versioning at petabyte scale. Provides Git-like branching
  for data lakes. Good for organizations with many teams sharing data.

---

## DVC in CI/CD

In our GitHub Actions workflows, DVC plays a critical role:

```yaml
# From train.yml - the training workflow
- name: Run data ingestion
  run: python -m src.data.ingest      # DVC stage: ingest

- name: Run data validation
  run: python -m src.data.validate    # DVC stage: validate

- name: Run preprocessing
  run: python -m src.data.preprocess  # DVC stage: preprocess
```

In CI/CD, we run the DVC stages directly (not via `dvc repro`) because the CI environment
starts fresh each time. However, we could also use `dvc pull` to restore previously cached
data and then `dvc repro` to only run changed stages:

```yaml
# Alternative CI/CD approach with DVC caching
- name: Setup DVC
  run: |
    pip install dvc[s3]
    dvc remote add -d storage s3://mlops-fraud-detection-011015903780
- name: Pull cached data
  run: dvc pull
- name: Reproduce pipeline
  run: dvc repro
```

---

## Interview Questions and Answers

### Q1: Why can't you just use Git to version your training data?

**A:** Git stores full copies of files and is designed for text-based diffs. A 150 MB CSV
committed 100 times would bloat the Git repository to 15 GB, making cloning painfully slow.
Git also does not support pipeline orchestration, metrics tracking, or remote storage
backends like S3. DVC solves all of these by storing only lightweight hash pointers in Git
and managing the actual data in cheap object storage.

### Q2: Explain how DVC achieves reproducibility.

**A:** DVC achieves reproducibility through three mechanisms: (1) Content-addressable storage
--- every file version is identified by its MD5 hash. Given the same hash, you get the exact
same data. (2) Pipeline DAG (dvc.yaml) --- captures the exact commands, dependencies, and
parameters for each step. (3) Lock file (dvc.lock) --- records the exact hashes of all
inputs and outputs from the last pipeline run. Together, checking out a specific Git commit
(which includes the DVC metadata) and running `dvc pull` + `dvc repro` will reproduce the
exact same model.

### Q3: What happens when you run `dvc repro` and nothing has changed?

**A:** DVC compares the current hashes of all dependencies (code files, data files,
parameters) with the hashes recorded in `dvc.lock`. If nothing has changed, DVC prints
"Stage 'X' didn't change, skipping" for every stage and completes in seconds. It does not
rerun any computation. This is possible because DVC uses content hashing, not timestamps.

### Q4: How does DVC handle large files that don't fit in memory?

**A:** DVC does not load files into memory. It computes MD5 hashes by streaming the file
in chunks. The actual data management (caching, pushing, pulling) is file-level, not
content-level. DVC treats files as opaque blobs --- it does not parse CSVs or understand
their contents. This means DVC can handle files of any size, limited only by disk space
and network bandwidth.

### Q5: What is the difference between `dvc push` and `git push`?

**A:** `git push` sends code, DVC metadata (`.dvc` files, `dvc.yaml`, `dvc.lock`), metrics,
and parameters to GitHub. `dvc push` sends the actual data files and model artifacts to the
configured remote storage (S3 in our case). Both commands are needed to fully share your
work with teammates. `git push` without `dvc push` means teammates get the metadata but
cannot download the actual data. `dvc push` without `git push` means the data is in S3 but
nobody has the metadata to find it.

### Q6: How would you handle data versioning for a dataset that changes daily?

**A:** For daily-changing data, I would: (1) Use DVC with a date-based tagging strategy
(Git tags like `data-2024-01-15`). (2) Set up a scheduled pipeline that runs `dvc add` on
the new data and commits the updated `.dvc` file. (3) Use `dvc diff` to compare data across
versions. (4) Implement a data validation step (like our pandera schema check) that runs
before any training to catch corrupted or anomalous data. (5) Set retention policies on the
S3 remote (our Terraform config expires old drift reports after 365 days and transitions
old models to S3-IA after 90 days).

### Q7: What are the limitations of DVC?

**A:** DVC has several limitations: (1) No built-in data-level versioning --- it tracks files,
not individual rows or columns. Changing one row in a 1 GB CSV creates a new 1 GB version.
(2) No native support for streaming data or real-time pipelines. (3) MD5 hashing can be
slow for very large files (100+ GB), though this is a one-time cost per version. (4) No
built-in access control --- anyone with S3 access can read all data. (5) The smart caching
(via `dvc repro`) does not work well if your pipeline stages have side effects or non-deterministic
behavior. (6) Merge conflicts in `.dvc` files can be confusing to resolve.

### Q8: Explain the difference between `dvc add` and defining a stage in `dvc.yaml`.

**A:** `dvc add` is for tracking files that are produced outside the DVC pipeline (e.g., data
downloaded manually or produced by an external system). It creates a `.dvc` file. Defining a
stage in `dvc.yaml` is for tracking files that are produced by your pipeline. The output is
declared in the `outs:` section of the stage. The key difference is that `dvc.yaml` stages
can be rerun with `dvc repro`, while `dvc add` files cannot --- they are static snapshots.
In our project, we use `dvc.yaml` stages for everything because the entire pipeline is
automated, from ingestion to evaluation.

---

## Practical Tips

1. **Always commit `dvc.lock` to Git.** It is the reproducibility contract. Without it,
   `dvc repro` cannot determine which stages need to rerun.

2. **Use `cache: false` for metrics files.** This keeps them in Git directly so you can
   use `dvc metrics diff` without pulling from the remote. Our `eval_metrics.json`,
   `confusion_matrix.json`, and `roc_curve.json` all use `cache: false`.

3. **Declare fine-grained parameter dependencies.** Instead of `params: - model` (which
   triggers rerun on ANY model param change), you can write `params: - model.params.max_depth`
   to only rerun when that specific param changes. Use broader dependencies (like we do)
   when parameters are tightly coupled.

4. **Run `dvc status` before `dvc repro`.** It shows which stages are outdated and why,
   without running anything. Useful for understanding what will happen.

5. **Use `dvc gc` periodically.** DVC's cache grows over time. `dvc gc --workspace` removes
   cached files not referenced by the current workspace, freeing disk space.

6. **Set up `.gitignore` correctly.** DVC automatically adds tracked files to `.gitignore`
   when you use `dvc add`. For pipeline outputs (defined in `dvc.yaml`), you need to add
   them to `.gitignore` yourself. Our `.gitignore` includes `data/raw/*.csv`,
   `data/processed/*.csv`, `data/processed/*.pkl`, and `models/*.pkl`.
