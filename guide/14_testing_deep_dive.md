# Guide 14: Testing Deep Dive -- Unit, Integration, and Model Quality Tests

## Table of Contents
1. [Why Testing ML Code is Different](#1-why-testing-ml-code-is-different)
2. [Pytest Fundamentals -- Everything Used in This Project](#2-pytest-fundamentals)
3. [Unit Tests -- Line by Line Explanation](#3-unit-tests)
4. [Integration Tests -- Line by Line](#4-integration-tests)
5. [Model Quality Tests -- The CI/CD Gate](#5-model-quality-tests)
6. [Testing Best Practices for ML](#6-testing-best-practices-for-ml)

---

## 1. Why Testing ML Code is Different

### Traditional Software vs ML Software

**Traditional software** is deterministic. If you write `add(2, 3)`, it returns 5 every
time. If it returns 6, the function is broken. Tests are binary: pass or fail, correct
or incorrect.

**ML software** is probabilistic. A fraud detection model might correctly flag 85% of
fraud. Is that good? It depends. You cannot test "is this prediction correct?" for a
single sample the way you test `add(2, 3)`. Instead, you test statistical properties:
"Does the model's recall exceed 0.80 on the test set?"

This fundamental difference creates unique testing challenges:

1. **Non-determinism:** Random seeds, GPU floating-point operations, and data shuffling
   can produce slightly different results across runs.
2. **Data dependency:** Tests must account for data quality, distribution, and format.
3. **Emergent behavior:** A model's behavior emerges from training, not from explicit
   programming. You cannot unit-test a learned decision boundary.
4. **Slow feedback:** Training a model takes minutes to hours. You cannot afford to
   retrain in every test.
5. **Multiple correctness criteria:** A model might have great accuracy but terrible
   fairness, or great precision but terrible recall.

### The Testing Pyramid for ML

```
                    /\
                   /  \
                  / A/B \         <- Production experiments
                 / Tests \           (canary, shadow mode)
                /----------\
               /  Model     \     <- Does the model meet quality bars?
              /  Quality     \       (AUC > 0.90, recall > threshold)
             /  Tests         \
            /------------------\
           /  Integration       \  <- Do components work together?
          /  Tests               \    (pipeline end-to-end)
         /------------------------\
        /  Unit Tests              \  <- Do individual functions work?
       /  (data transforms, utils)  \    (preprocessing, feature eng.)
      /------------------------------\
```

**Unit tests** are fast (milliseconds), numerous, and test individual functions in
isolation. Example: "Does the scaler standardize the Amount column?"

**Integration tests** verify that components work together. Example: "Does the full
ingest-validate pipeline produce data with the correct shape and both classes?"

**Model quality tests** check that the trained model meets business requirements.
Example: "Is AUC-ROC above 0.90?" These are slower because they depend on a trained
model.

**A/B tests** (production experiments) verify that a model improves real-world outcomes.
These are not automated tests in the traditional sense but are the ultimate validation.

### What to Test

| Layer | What | Example |
|-------|------|---------|
| Data | Schema, types, ranges, nulls | "Class column has only 0 and 1" |
| Transforms | Preprocessing functions | "StandardScaler produces mean near 0" |
| Model behavior | Outputs, probabilities | "predict_proba sums to 1.0" |
| Model quality | Business metrics | "AUC-ROC >= 0.90" |
| Pipeline | End-to-end flow | "ingest -> validate produces 31 columns" |
| API | Request/response contracts | "POST /predict returns 200 with valid JSON" |

### Real Example: The $2M Preprocessing Bug

A fintech company had a fraud model in production. A developer changed the preprocessing
code to handle a new feature but accidentally broke the scaling logic for the "Amount"
feature. Instead of standardizing Amount (mean=0, std=1), the code passed raw dollar
amounts to the model.

The model had been trained on standardized amounts (values like -0.5, 0.0, 1.2). Now
it was receiving raw amounts ($15.99, $459.00, $2,300.00). The model interpreted these
as extreme outliers and flagged almost every high-value transaction as fraud.

The result: 80% of transactions over $200 were declined for two days. Customer
complaints surged. Estimated revenue loss: $2M.

**What would have caught this:**
- A unit test checking that Amount's mean is approximately 0 and std is approximately 1
  after preprocessing (exactly like `test_scaling_standardizes_amount` in this project).
- An integration test checking that the preprocessed data has expected statistical
  properties.
- A data validation step in the pipeline (like Great Expectations).

---

## 2. Pytest Fundamentals -- Everything Used in This Project

### The `pyproject.toml` Configuration

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

**Line by line:**

**`testpaths = ["tests"]`** -- tells pytest where to look for test files. Without this,
pytest searches the entire project directory. Setting it explicitly:
- Speeds up test discovery (does not scan `src/`, `data/`, `metrics/`).
- Prevents accidentally picking up test-like files in non-test directories.

**`addopts = "-v --tb=short"`** -- default command-line options appended to every pytest
invocation:
- `-v` (verbose): shows each test name and its pass/fail status individually, instead of
  just a summary. Output looks like:
  ```
  tests/unit/test_preprocess.py::TestPreprocessing::test_no_nulls_after_preprocessing PASSED
  tests/unit/test_preprocess.py::TestPreprocessing::test_class_column_is_binary PASSED
  ```
- `--tb=short`: when a test fails, show a short traceback instead of the full one. Full
  tracebacks include every frame in the call stack and can be overwhelming. Short
  tracebacks show just the failing assertion and the immediate context.

**`markers`** -- declares custom markers that categorize tests. Each entry is
`"marker_name: description"`. Without declaring markers, pytest shows a warning when you
use `@pytest.mark.unit` because it does not know if you misspelled a built-in marker.

**How markers work with the `-m` flag:**
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only model quality tests
pytest -m model

# Run everything except model tests (fast CI feedback)
pytest -m "not model"

# Run unit AND integration tests
pytest -m "unit or integration"
```

This is critical for CI/CD pipeline design:
- **On every commit:** run `pytest -m unit` (fast, seconds).
- **On pull request:** run `pytest -m "unit or integration"` (medium, seconds to minutes).
- **After training:** run `pytest -m model` (requires trained model artifacts).
- **Full suite:** run `pytest` with no `-m` filter (everything).

### Test Class Organization

```python
@pytest.mark.unit
class TestPreprocessing:
    ...
```

**Why classes vs standalone functions?**

Pytest supports both styles:

**Class-based (used in this project):**
```python
class TestPreprocessing:
    def test_no_nulls(self):
        ...
    def test_class_is_binary(self):
        ...
```

**Function-based (alternative):**
```python
def test_no_nulls():
    ...
def test_class_is_binary():
    ...
```

**Why this project uses classes:**

1. **Logical grouping:** `TestPreprocessing` groups all preprocessing tests. `TestModel`
   groups all model tests. When running pytest with `-v`, the output shows the class
   name, making it easy to see which component failed.

2. **Shared helpers:** The `_make_sample_data` method can be called by any test method in
   the class via `self._make_sample_data()`. With functions, you would need module-level
   fixtures or helper functions.

3. **Marker inheritance:** The `@pytest.mark.unit` decorator on the class applies to ALL
   methods in the class. Without classes, you would need to decorate every function
   individually.

4. **No `__init__` needed:** Pytest test classes should NOT have `__init__` methods.
   Pytest creates a new instance of the class for each test method, ensuring test
   isolation.

**Important:** The `self` parameter in test methods is just the test class instance, not
a fixture. Unlike `unittest.TestCase`, pytest test classes do not inherit from any base
class.

### Helper Methods -- `_make_sample_data`

```python
def _make_sample_data(self, n: int = 1000) -> pd.DataFrame:
    ...
```

The leading underscore `_` is a Python convention meaning "this is an internal/private
method." Pytest only collects methods that start with `test_`. Methods starting with `_`
are ignored by the test runner, making them available as utilities.

### `pytest.skip()` -- Conditional Skipping

```python
if not path.exists():
    pytest.skip("eval_metrics.json not found -- run evaluate first")
```

`pytest.skip()` immediately stops the current test and marks it as "skipped" rather than
"passed" or "failed." The message explains why.

**Output:**
```
test_model_quality.py::TestModelQuality::test_auc_roc_above_threshold SKIPPED
    (eval_metrics.json not found -- run evaluate first)
```

**When to use skip:**
- The test depends on an artifact that may not exist yet (model file, metrics file).
- The test requires a specific environment (GPU, specific OS, API credentials).
- The test is temporarily broken and you want to track it without failing the suite.

**skip vs xfail:**
- `pytest.skip()` -- "this test cannot run right now."
- `@pytest.mark.xfail` -- "this test is expected to fail" (useful for known bugs).

### `pytest.raises()` -- Testing Error Conditions

```python
with pytest.raises(Exception):
    validate_data(df)
```

This context manager asserts that the code inside the `with` block raises an exception.
If no exception is raised, the test FAILS. If the specified exception (or a subclass)
is raised, the test PASSES.

**Why test errors?** You need to verify that your code fails correctly. If `validate_data`
receives data with `Class=5` (invalid), it MUST raise an exception. If it silently
accepts bad data, the model will train on garbage.

**More specific usage (not in this project but good to know):**
```python
with pytest.raises(ValueError, match="Class column contains invalid values"):
    validate_data(df)
```
This checks both the exception type AND the message pattern.

### Code Coverage

The project does not include `--cov` in `addopts`, but in a CI/CD pipeline you would
run:

```bash
pytest --cov=src --cov-report=term-missing
```

**`--cov=src`** -- measure code coverage for the `src/` package. Coverage means: what
percentage of lines in `src/` were executed during the test run?

**`--cov-report=term-missing`** -- print a terminal report showing which lines were NOT
covered. Output:
```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/data/ingest.py                   35      2    94%   45-46
src/data/preprocess.py               52      8    85%   23-25, 67-71
src/monitoring/drift_detection.py    68     25    63%   82-141
```

**What is good coverage for ML projects?**
- Data processing code: aim for 85%+. This code is deterministic and testable.
- Model training code: 60-80% is realistic. Hard to test every training path.
- Monitoring/deployment code: 50-70%. Often depends on external services (AWS, etc.).
- Overall: 70%+ is a reasonable target. 100% is not practical for ML code.

---

## 3. Unit Tests -- Line by Line Explanation

### test_preprocess.py -- Complete Walkthrough

**Source file:** `tests/unit/test_preprocess.py`

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
```

**Imports:**
- `numpy` -- for random number generation and numerical comparisons.
- `pandas` -- for creating DataFrames that simulate real data.
- `pytest` -- the test framework. Used for markers in this file.
- `StandardScaler` -- the same scaler used in the actual preprocessing code. We import
  it here to verify its behavior in isolation.

```python
@pytest.mark.unit
class TestPreprocessing:
```

The `@pytest.mark.unit` marker means all tests in this class can be run with
`pytest -m unit`. This class tests data preprocessing operations -- the first step
of the ML pipeline.

#### `_make_sample_data` -- Creating Controlled Test Data

```python
def _make_sample_data(self, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "Time": rng.uniform(0, 172800, n),
        "Amount": rng.exponential(100, n),
        "Class": rng.choice([0, 1], n, p=[0.99, 0.01]),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    return pd.DataFrame(data)
```

**`rng = np.random.default_rng(42)`** -- creates a random number generator with seed 42.

**Why `default_rng(42)` instead of `np.random.seed(42)`?**
- `np.random.seed()` sets a global random state. If two tests run in the same process,
  one test's randomness affects the other. This creates order-dependent test results.
- `default_rng(42)` creates an isolated generator. Each call to `_make_sample_data`
  creates its own generator with the same seed, producing identical data regardless of
  what other tests do. This is the modern NumPy approach (introduced in NumPy 1.17).

**Why seed 42?** It is a convention (a reference to "The Hitchhiker's Guide to the
Galaxy"). Any fixed seed works. The point is reproducibility: every test run generates
the exact same data.

**`"Time": rng.uniform(0, 172800, n)`** -- simulates the Time feature.
- `uniform(0, 172800)` generates random floats between 0 and 172800 (seconds in 2 days).
- In the real credit card dataset, Time is the seconds elapsed since the first
  transaction in the dataset.

**`"Amount": rng.exponential(100, n)`** -- simulates the Amount feature.
- `exponential(100)` generates from an exponential distribution with mean $100.
- Exponential is a good approximation for transaction amounts: many small transactions,
  few large ones. The distribution is right-skewed, just like real spending data.

**`"Class": rng.choice([0, 1], n, p=[0.99, 0.01])`** -- simulates the target variable.
- `choice([0, 1], n, p=[0.99, 0.01])` randomly picks 0 or 1, with 99% probability of 0
  (legitimate) and 1% probability of 1 (fraud).
- This mimics the real dataset's class imbalance (0.17% fraud rate). Using 1% instead of
  0.17% ensures the test data has enough fraud samples (about 10 out of 1000) to be
  meaningful.

**`for i in range(1, 29): data[f"V{i}"] = rng.standard_normal(n)`** -- generates V1
through V28.
- `standard_normal(n)` generates from N(0,1) -- standard normal distribution.
- In the real dataset, V1-V28 are PCA-transformed features that are already approximately
  normally distributed. Using standard normal is a reasonable simulation.

**`return pd.DataFrame(data)`** -- assembles the dictionary into a DataFrame with 31
columns: Time, V1-V28, Amount, Class. This matches the real dataset's schema.

**Design principle:** The test data is synthetic but structurally identical to real data.
Tests validate behavior on this controlled data, ensuring preprocessing code handles the
expected schema correctly.

#### `test_no_nulls_after_preprocessing`

```python
def test_no_nulls_after_preprocessing(self):
    df = self._make_sample_data()
    df = df.drop(columns=["Time"])
    assert df.isnull().sum().sum() == 0
```

**What this tests:** After dropping the Time column (which the real preprocessing does),
the resulting DataFrame has zero null values.

**Line by line:**
- `df = self._make_sample_data()` -- creates 1000 rows of synthetic data.
- `df = df.drop(columns=["Time"])` -- drops the Time column, simulating the preprocessing
  step. Time is dropped because it is a sequential identifier, not a predictive feature.
- `df.isnull()` -- returns a boolean DataFrame where True means null/NaN.
- `.sum()` -- sums each column (True=1, False=0), giving null counts per column.
- `.sum()` -- sums across columns, giving the total null count.
- `assert ... == 0` -- asserts zero total nulls.

**Why this test matters:** Null values in features cause XGBoost to handle them implicitly
(it has built-in null handling), which can lead to unexpected behavior. Worse, nulls in
other ML frameworks (logistic regression, neural networks) cause crashes. This test
ensures the preprocessing pipeline eliminates nulls, making the data safe for any
downstream model.

**In production:** If a data pipeline starts producing nulls (e.g., a database column
becomes nullable due to a schema migration), this test catches it before the model
sees garbage.

#### `test_class_column_is_binary`

```python
def test_class_column_is_binary(self):
    df = self._make_sample_data()
    assert set(df["Class"].unique()).issubset({0, 1})
```

**What this tests:** The Class column contains only 0 and 1, nothing else.

**Line by line:**
- `df["Class"].unique()` -- returns the unique values in the Class column as a numpy
  array. For our test data, this should be `array([0, 1])`.
- `set(...)` -- converts to a Python set for set operations. Result: `{0, 1}`.
- `.issubset({0, 1})` -- checks if the unique values are a subset of `{0, 1}`. This
  returns True if Class contains only 0s, only 1s, or both 0s and 1s.

**Why `issubset` instead of `== {0, 1}`?** If the test data (by random chance with a
very small n) happened to contain only class 0, `== {0, 1}` would fail even though the
data is valid. `issubset` is more lenient: "every value is either 0 or 1" is the
correct invariant.

**Why this test matters:** If Class contained values like 2, 5, or -1, the binary
classifier would either crash or produce meaningless results. If Class contained float
values like 0.5, the classification metrics would be wrong. This test guards against
upstream data corruption.

#### `test_scaling_standardizes_amount`

```python
def test_scaling_standardizes_amount(self):
    df = self._make_sample_data()
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    assert abs(df["Amount"].mean()) < 0.1
    assert abs(df["Amount"].std() - 1.0) < 0.1
```

**What this tests:** After applying StandardScaler to the Amount column, the resulting
values have approximately mean 0 and standard deviation 1.

**Line by line:**
- `scaler = StandardScaler()` -- creates a new StandardScaler instance.
  StandardScaler transforms data using: `z = (x - mean) / std`.
- `df[["Amount"]]` -- double brackets return a DataFrame (2D), not a Series (1D).
  `fit_transform` expects 2D input. `df["Amount"]` would return a 1D Series and cause
  an error.
- `scaler.fit_transform(df[["Amount"]])` -- fits the scaler (computes mean and std of
  Amount) and transforms in one step. Returns a 2D numpy array.
- `df["Amount"] = ...` -- replaces the Amount column with the scaled values.
- `abs(df["Amount"].mean()) < 0.1` -- the mean should be near 0. We use a tolerance of
  0.1 instead of exact 0 because:
  - Floating-point arithmetic introduces tiny rounding errors.
  - With finite samples, the mean after standardization is approximately but not exactly
    zero.
- `abs(df["Amount"].std() - 1.0) < 0.1` -- the std should be near 1.0. Same reasoning
  for the tolerance.

**Why not use exact equality (== 0)?** In ML testing, approximate assertions are the
norm. `assert df["Amount"].mean() == 0.0` would almost always fail due to floating-point
precision. The tolerance of 0.1 is generous but appropriate for validating the
transformation is working correctly (not that it is perfect to 15 decimal places).

**Why this test matters:** This is exactly the test that would have prevented the $2M
bug described earlier. If someone breaks the scaling logic, this test catches it before
the code is merged.

#### `test_train_test_split_preserves_ratio`

```python
def test_train_test_split_preserves_ratio(self):
    from sklearn.model_selection import train_test_split

    df = self._make_sample_data(n=10000)
    y = df["Class"]
    _, _, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.01
```

**What this tests:** When splitting data with stratification, the fraud ratio in the
training set matches the fraud ratio in the test set.

**Line by line:**
- `from sklearn.model_selection import train_test_split` -- imported inside the method,
  not at the top of the file. This is an unusual pattern. It works because Python caches
  imports, so the performance cost is negligible. The import is here because it is only
  needed by this one test.

- `self._make_sample_data(n=10000)` -- uses 10,000 samples instead of the default 1,000.
  Why? With 1% fraud rate and 1,000 samples, you get roughly 10 fraud cases. After an
  80/20 split, the test set has approximately 2 fraud cases. The ratio of 2 fraud out of
  200 total (1.0%) vs 8 fraud out of 800 total (1.0%) could easily differ by more than
  0.01 just from rounding. With 10,000 samples, you get about 100 fraud cases, making
  the ratio much more stable.

- `train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)` -- splits into
  80% train, 20% test.
  - `stratify=y` is the key parameter: it ensures both splits have the same proportion
    of each class. Without stratification, a random split might put all 10 fraud cases
    in the training set and none in the test set.
  - `random_state=42` makes the split reproducible.
  - Returns four values: `X_train, X_test, y_train, y_test`. The `_, _` discards the
    X arrays since we only need the y arrays to check class ratios.

- `y_train.mean()` -- since fraud is 1 and legitimate is 0, the mean equals the fraud
  ratio. If 80 out of 8000 training samples are fraud, the mean is 0.01 (1%).

- `abs(train_ratio - test_ratio) < 0.01` -- the fraud ratio in train and test should be
  within 1 percentage point of each other. With stratification, they should be nearly
  identical.

**Why this test matters:** Without stratification in an imbalanced dataset, the test set
might have zero fraud cases, making evaluation meaningless. This test ensures the
splitting preserves class balance.

#### `test_feature_count_after_drop`

```python
def test_feature_count_after_drop(self):
    df = self._make_sample_data()
    df = df.drop(columns=["Time", "Class"])
    assert df.shape[1] == 29  # V1-V28 + Amount
```

**What this tests:** After dropping Time and Class, exactly 29 feature columns remain.

**Line by line:**
- `df.drop(columns=["Time", "Class"])` -- drops two columns. Time is not a feature.
  Class is the target (not a feature for the model).
- `df.shape[1]` -- the number of columns. `shape` returns `(n_rows, n_cols)`.
- `== 29` -- V1 through V28 (28 columns) plus Amount (1 column) = 29 features.

**Why this test matters:** If someone accidentally drops an extra column or fails to
drop Time, the feature count changes. A model trained on 29 features will crash if
given 28 or 30 features at inference time. This is a shape invariant test.

**The comment `# V1-V28 + Amount` is important.** It explains the magic number 29. Without
it, a future developer might wonder why 29 is the expected count.

---

### test_model.py -- Complete Walkthrough

**Source file:** `tests/unit/test_model.py`

```python
import numpy as np
import pytest
import xgboost as xgb
```

**Imports:**
- `xgboost as xgb` -- the XGBoost library. We import it to train a small model for
  testing. Note: this tests XGBoost's behavior and our expectations of it, not our
  training code directly.

```python
@pytest.mark.unit
class TestModel:
```

Despite using a trained model, these are unit tests because they test model behavior
properties in isolation, not the full training pipeline.

#### `_train_dummy_model` -- Why Train a Small Model for Tests?

```python
def _train_dummy_model(self):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 10))
    y = rng.choice([0, 1], 500, p=[0.95, 0.05])
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y, verbose=False)
    return model
```

**Why not use the real trained model?**
1. **Speed:** The real model might have 200 trees of depth 8, taking seconds to load.
   This dummy model has 10 trees of depth 3, training in milliseconds.
2. **Independence:** Unit tests should not depend on external artifacts (model files).
   If the model file is missing or corrupted, unit tests should still run.
3. **Reproducibility:** The dummy model is fully deterministic (fixed seed, fixed data).
   The real model depends on the training data, which might change.

**Line by line:**
- `X = rng.standard_normal((500, 10))` -- 500 samples, 10 features. Small enough to
  train instantly, large enough to produce a non-trivial model.
- `y = rng.choice([0, 1], 500, p=[0.95, 0.05])` -- 5% positive rate. Mimics the
  imbalanced nature of fraud data.
- `n_estimators=10` -- only 10 trees (the real model might use 100-500).
- `max_depth=3` -- shallow trees (the real model might use depth 6-8).
- `random_state=42` -- ensures the model itself is reproducible.
- `verbose=False` -- suppresses XGBoost's training output. Without this, every test
  run would print training progress, cluttering the test output.

#### `test_model_outputs_probabilities`

```python
def test_model_outputs_probabilities(self):
    model = self._train_dummy_model()
    rng = np.random.default_rng(99)
    X = rng.standard_normal((10, 10))
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
```

**What this tests:** The model's `predict_proba` returns an array with the correct shape
where each row sums to 1.0.

**Line by line:**
- `np.random.default_rng(99)` -- uses a DIFFERENT seed (99) than the training data (42).
  This ensures we are testing on data the model has never seen, not accidentally testing
  on training data.
- `rng.standard_normal((10, 10))` -- 10 test samples, 10 features (matching the training
  feature count).
- `model.predict_proba(X)` -- returns a 2D array of shape (n_samples, n_classes).
  Each row is [P(class=0), P(class=1)].

- `assert probs.shape == (10, 2)` -- shape check.
  - 10 rows: one probability vector per input sample.
  - 2 columns: one per class (binary classification). If the model were 3-class, this
    would be (10, 3).
  - **Why this matters:** If the model returns the wrong shape, downstream code that
    expects `probs[:, 1]` (fraud probability) will either crash or return wrong values.

- `np.allclose(probs.sum(axis=1), 1.0)` -- checks that probabilities sum to 1.0 for
  each sample.
  - `probs.sum(axis=1)` sums across columns for each row. Result: array of 10 values,
    each should be 1.0.
  - `np.allclose(..., 1.0)` checks approximate equality with default tolerances
    (atol=1e-8, rtol=1e-5). This is better than `==` because floating-point
    probabilities might sum to 0.9999999999 instead of exactly 1.0.

**Why this test matters:** This verifies a fundamental property of probability outputs.
If `predict_proba` returned unnormalized scores (which some frameworks can do), threshold-
based decisions would be wrong. A threshold of 0.5 assumes probabilities, not raw scores.

#### `test_model_outputs_binary_predictions`

```python
def test_model_outputs_binary_predictions(self):
    model = self._train_dummy_model()
    rng = np.random.default_rng(99)
    X = rng.standard_normal((10, 10))
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
```

**What this tests:** The model's `predict` method returns only 0s and 1s.

**Line by line:**
- `model.predict(X)` -- returns hard predictions (class labels), not probabilities.
- `np.unique(preds)` -- gets the unique predicted values.
- `set(...).issubset({0, 1})` -- same pattern as the Class column test. Verifies the
  output domain is {0, 1}, not {0, 1, 2} or {-1, 1} or continuous values.

**Why `issubset` instead of `== {0, 1}`?** With only 10 test samples and 95% class 0,
the model might predict all 0s (no fraud). `{0}.issubset({0, 1})` is True, which is
correct -- the outputs are still valid binary predictions.

#### `test_model_deterministic`

```python
def test_model_deterministic(self):
    model = self._train_dummy_model()
    rng = np.random.default_rng(99)
    X = rng.standard_normal((10, 10))
    pred1 = model.predict_proba(X)
    pred2 = model.predict_proba(X)
    np.testing.assert_array_equal(pred1, pred2)
```

**What this tests:** Given the same input, the model produces exactly the same output
every time.

**Line by line:**
- Call `predict_proba` twice with the exact same input `X`.
- `np.testing.assert_array_equal(pred1, pred2)` -- asserts that every element in `pred1`
  equals the corresponding element in `pred2`. This is EXACT equality (not approximate).

**Why exact equality is appropriate here:** Unlike training (which might involve random
initialization), inference is deterministic for tree-based models like XGBoost. There
is no randomness in traversing a decision tree. If inference were non-deterministic,
it would indicate a serious bug (thread-safety issue, uninitialized memory, etc.).

**`np.testing.assert_array_equal` vs regular `assert`:**
- `assert np.array_equal(pred1, pred2)` would work but gives an unhelpful error message
  on failure: `AssertionError`.
- `np.testing.assert_array_equal` gives a detailed error showing which elements differ:
  ```
  AssertionError:
  Arrays are not equal
  Mismatched elements: 3 / 20 (15.0%)
  Max absolute difference: 0.023
  x: array([[0.92, 0.08], [0.95, 0.05], ...])
  y: array([[0.92, 0.08], [0.93, 0.07], ...])
  ```
  This makes debugging much easier.

**Why this test matters:** Determinism is essential for:
- Debugging: you can reproduce a specific prediction.
- Compliance: regulators may require explanations for specific decisions.
- Testing: non-deterministic predictions make other tests flaky.

#### `test_model_handles_single_sample`

```python
def test_model_handles_single_sample(self):
    model = self._train_dummy_model()
    X = np.random.default_rng(99).standard_normal((1, 10))
    pred = model.predict(X)
    assert len(pred) == 1
```

**What this tests:** The model can handle a single sample (batch size 1).

**Line by line:**
- `standard_normal((1, 10))` -- a single row with 10 features. Shape: `(1, 10)`.
- `model.predict(X)` -- predict on one sample.
- `assert len(pred) == 1` -- exactly one prediction returned.

**Why this is an edge case worth testing:**
1. In production, the model might receive one request at a time (real-time serving).
   During training and evaluation, the model always sees batches.
2. Some frameworks have bugs with batch size 1. For example, if the code accidentally
   does `X.reshape(-1)` instead of `X.reshape(1, -1)`, the model receives a 1D array
   of 10 features instead of a 2D array of 1 sample with 10 features. This can cause
   subtle shape errors.
3. Edge cases are where bugs hide. Testing batch size 1, empty input, and very large
   batches covers the boundary conditions.

---

## 4. Integration Tests -- Line by Line

**Source file:** `tests/integration/test_pipeline.py`

```python
import pytest

from src.data.ingest import generate_synthetic_fraud_data
from src.data.validate import validate_data
```

**Imports:**
- `generate_synthetic_fraud_data` -- the data ingestion function that creates synthetic
  fraud data.
- `validate_data` -- the data validation function that checks data quality.

These are **real functions from the codebase**, not mocks. Integration tests exercise
the actual code paths, unlike unit tests that might mock dependencies.

```python
@pytest.mark.integration
class TestPipeline:
```

Marked as integration tests. These test that multiple components work together correctly.

#### `test_ingest_generates_correct_shape`

```python
def test_ingest_generates_correct_shape(self):
    df = generate_synthetic_fraud_data(n_samples=1000)
    assert df.shape[0] == 1000
    assert df.shape[1] == 31  # Time + V1-V28 + Amount + Class
```

**What this tests:** The ingestion function produces a DataFrame with the correct
dimensions.

- `df.shape[0] == 1000` -- exactly 1000 rows as requested.
- `df.shape[1] == 31` -- exactly 31 columns: Time (1) + V1-V28 (28) + Amount (1) +
  Class (1) = 31.

**Why this is an integration test, not a unit test:**
It tests the full output of `generate_synthetic_fraud_data`, which internally may
create multiple features, combine them, and return a DataFrame. It verifies the
end-to-end contract of the function.

**The comment `# Time + V1-V28 + Amount + Class` is essential.** Without it, the magic
number 31 is meaningless. Comments explaining magic numbers prevent confusion when
someone later asks "why 31?"

#### `test_ingest_generates_fraud_and_legit`

```python
def test_ingest_generates_fraud_and_legit(self):
    df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.01)
    assert df["Class"].sum() > 0
    assert (df["Class"] == 0).sum() > 0
```

**What this tests:** The generated data contains both fraud (Class=1) and legitimate
(Class=0) transactions.

- `n_samples=10000, fraud_ratio=0.01` -- requests 10,000 samples with 1% fraud.
  Uses 10,000 (not 100) to ensure the expected 100 fraud cases are not zero by chance.
- `df["Class"].sum() > 0` -- at least one fraud case exists. Since Class is 0/1, the
  sum equals the count of fraud cases.
- `(df["Class"] == 0).sum() > 0` -- at least one legitimate case exists.

**Why this test matters:** A data generation bug could create all-fraud or all-legitimate
data. The model would then:
- All fraud: learn to always predict fraud. Precision = fraud_ratio (terrible).
- All legitimate: learn to never predict fraud. Recall = 0 (catastrophic).

This test is a sanity check that the data generation respects the fraud_ratio parameter.

#### `test_validate_passes_on_clean_data`

```python
def test_validate_passes_on_clean_data(self):
    df = generate_synthetic_fraud_data(n_samples=1000)
    validated = validate_data(df)
    assert len(validated) == 1000
```

**What this tests:** The validation function accepts clean (correctly generated) data
without raising exceptions and returns all rows.

This is a **happy path** test. It verifies that the normal case works before testing
error cases.

- `validated = validate_data(df)` -- runs validation. If this raises an exception,
  the test fails.
- `assert len(validated) == 1000` -- no rows were dropped. Validation either accepts
  the data (returns it) or rejects it (raises an exception). With clean data, it
  should accept everything.

**Why test the happy path?** It seems obvious, but without this test:
- A bug in `validate_data` that rejects valid data would go unnoticed.
- A change to `generate_synthetic_fraud_data` that produces slightly different but still
  valid data would trigger a false alarm in validation.

#### `test_validate_rejects_bad_data`

```python
def test_validate_rejects_bad_data(self):
    df = generate_synthetic_fraud_data(n_samples=100)
    df.loc[0, "Class"] = 5.0  # invalid class
    with pytest.raises(Exception):
        validate_data(df)
```

**What this tests:** The validation function raises an exception when given data with
an invalid Class value.

**Line by line:**
- `generate_synthetic_fraud_data(n_samples=100)` -- creates valid data first.
- `df.loc[0, "Class"] = 5.0` -- corrupts the first row by setting Class to 5.0 (an
  invalid value; Class should only be 0 or 1).
  - `df.loc[0, "Class"]` uses label-based indexing. Row 0, column "Class."
  - Setting to 5.0 (not 5) because the column is likely float64.
- `with pytest.raises(Exception):` -- expects that `validate_data(df)` will raise any
  type of Exception.

**Why `Exception` and not a more specific type?**
Using the broad `Exception` type is a pragmatic choice. The test verifies that SOME
error is raised, without coupling to the specific exception class. If `validate_data`
changes from raising `ValueError` to `ValidationError`, the test still passes. However,
in a more rigorous codebase, you would test for the specific exception type.

**Why this test matters:** This is the "negative path" test. If `validate_data` silently
accepts Class=5, that corrupted data flows into training, poisoning the model. Validation
is a gate -- it must reject bad data reliably.

---

## 5. Model Quality Tests -- The CI/CD Gate

**Source file:** `tests/model/test_model_quality.py`

```python
import json
import pytest
from src.utils.config import get_project_root
```

These tests are fundamentally different from unit and integration tests. They do not test
code logic; they test model output quality. They read persisted metrics from a file
produced by the evaluation step and check if those metrics meet business requirements.

```python
@pytest.mark.model
class TestModelQuality:
```

The `model` marker separates these tests from the faster unit/integration tests. These
tests only make sense after a model has been trained and evaluated. They are typically
run as a CI/CD gate before deployment.

#### `_load_metrics` -- Reading Persisted Evaluation Metrics

```python
def _load_metrics(self) -> dict:
    path = get_project_root() / "metrics" / "eval_metrics.json"
    if not path.exists():
        pytest.skip("eval_metrics.json not found -- run evaluate first")
    with open(path) as f:
        return json.load(f)
```

**Line by line:**
- `get_project_root() / "metrics" / "eval_metrics.json"` -- constructs the path to the
  metrics file using `pathlib`'s `/` operator. The metrics file is generated by the
  evaluation pipeline step.
- `if not path.exists(): pytest.skip(...)` -- if the file does not exist (model has not
  been evaluated yet), skip the test with an explanatory message. This prevents false
  failures when running the full test suite before training.
- `json.load(f)` -- loads the JSON into a Python dict. The expected structure:
  ```json
  {
      "auc_roc": 0.95,
      "recall": 0.80,
      "precision": 0.85,
      "f1": 0.82,
      "test_samples": 56962,
      "predicted_fraud_count": 45
  }
  ```

**Why read from a file instead of computing metrics in the test?**
1. **Separation of concerns:** The evaluation pipeline computes metrics. The test
   validates them. These are different responsibilities.
2. **Speed:** Computing metrics requires loading the model and test data (seconds to
   minutes). Reading a JSON file is instant.
3. **Reproducibility:** The JSON file is an artifact that can be versioned and audited.
4. **CI/CD pipeline design:** The evaluation step runs once, producing the metrics file.
   Multiple quality tests can then read the same file.

#### `test_auc_roc_above_threshold`

```python
def test_auc_roc_above_threshold(self):
    metrics = self._load_metrics()
    assert metrics["auc_roc"] >= 0.90, f"AUC-ROC {metrics['auc_roc']} below 0.90"
```

**What this tests:** The model's AUC-ROC is at least 0.90.

- `metrics["auc_roc"]` -- the AUC-ROC score from the evaluation.
- `>= 0.90` -- the minimum acceptable AUC-ROC. This is a business-defined threshold.
  An AUC of 0.90 means the model correctly ranks a random fraud case higher than a
  random legitimate case 90% of the time.
- The f-string error message `f"AUC-ROC {metrics['auc_roc']} below 0.90"` provides
  immediate context when the test fails: "AUC-ROC 0.87 below 0.90" is more useful than
  a bare `AssertionError`.

**Why 0.90?**
- AUC < 0.50 = worse than random.
- AUC 0.50-0.70 = poor, barely useful.
- AUC 0.70-0.80 = fair, acceptable for some use cases.
- AUC 0.80-0.90 = good.
- AUC 0.90-0.95 = very good.
- AUC 0.95-1.00 = excellent (possible overfitting if too high).

For fraud detection, 0.90 is a reasonable minimum. Below this, the model is not
discriminating well enough between fraud and legitimate transactions.

#### `test_recall_above_threshold`

```python
def test_recall_above_threshold(self):
    metrics = self._load_metrics()
    assert metrics["recall"] >= 0.02, f"Recall {metrics['recall']} below 0.02"
```

**What this tests:** The model catches at least 2% of fraud cases.

**Why recall is business-critical for fraud detection:**
Recall = (fraud cases caught) / (total actual fraud cases). A recall of 0.02 means the
model catches 2% of fraud. This seems low, but consider:
- In highly imbalanced datasets, a model might predict all-legitimate (recall = 0.0)
  and still have 99.83% accuracy. The threshold of 0.02 ensures the model is at least
  detecting SOME fraud.
- The actual production threshold would be much higher (0.60-0.80), but 0.02 is a
  minimum "the model is not completely broken" check.

**The consequence of zero recall:** Every fraud transaction goes undetected. The bank
loses money on every single fraud case. This test prevents deploying a degenerate model
that never predicts fraud.

#### `test_precision_above_threshold`

```python
def test_precision_above_threshold(self):
    metrics = self._load_metrics()
    assert metrics["precision"] >= 0.50, f"Precision {metrics['precision']} below 0.50"
```

**What this tests:** At least 50% of the transactions the model flags as fraud are
actually fraud.

Precision = (true positives) / (true positives + false positives). Precision of 0.50
means half the alerts are real fraud, half are false positives.

**Why 0.50?** If precision drops below 0.50, the fraud investigation team spends more
time on false alarms than on real fraud. This leads to:
- Alert fatigue (investigators start ignoring alerts).
- Wasted resources (each investigation costs time and money).
- Customer frustration (legitimate transactions get blocked).

#### `test_model_not_predicting_all_same_class`

```python
def test_model_not_predicting_all_same_class(self):
    metrics = self._load_metrics()
    assert metrics["predicted_fraud_count"] > 0, "Model predicts no fraud"
    total = metrics["test_samples"]
    assert metrics["predicted_fraud_count"] < total, "Model predicts all fraud"
```

**What this tests:** The model is making non-degenerate predictions -- it predicts SOME
fraud but not ALL fraud.

**Line by line:**
- `metrics["predicted_fraud_count"] > 0` -- the model predicts at least one fraud case.
  If the model predicts zero fraud, it has collapsed to the trivial "everything is
  legitimate" solution. This can happen with:
  - Extreme class imbalance (the model learns that predicting "legitimate" is always
    safe).
  - A bug in the decision threshold (threshold set to 1.0 instead of 0.5).
  - A training failure (model did not converge).

- `metrics["predicted_fraud_count"] < total` -- the model does not predict ALL
  transactions as fraud. This can happen with:
  - A broken feature (all inputs look the same, and the default prediction is fraud).
  - A bug in the threshold (threshold set to 0.0).
  - Data leakage (the model learned an artifact that correlates with fraud).

**Why two assertions instead of one?** Each failure mode has a different error message
and implies a different root cause. Knowing "model predicts no fraud" vs "model predicts
all fraud" immediately narrows the investigation.

**Real-world example -- catching a memorization bug:**
A team trained a model that achieved 0.999 AUC on the test set. Suspiciously perfect.
The `test_model_not_predicting_all_same_class` test passed, but further investigation
revealed that a timestamp feature was leaking future information. The model memorized
which transactions occurred near chargebacks. In production (where future information
is unavailable), the model performed at chance level. This led to adding a "feature
leakage" check: if AUC > 0.99, investigate for potential leakage.

### How These Tests Block Deployments in CI/CD

In the CI/CD pipeline (e.g., GitHub Actions), model quality tests serve as a deployment
gate:

```yaml
# .github/workflows/deploy.yml (simplified)
jobs:
  train:
    steps:
      - run: python -m src.models.train
      - run: python -m src.models.evaluate
      # This produces metrics/eval_metrics.json

  quality-gate:
    needs: train
    steps:
      - run: pytest -m model
      # If any quality test fails, the pipeline stops here.
      # The model is NOT deployed.

  deploy:
    needs: quality-gate
    steps:
      - run: python -m src.deployment.deploy_to_sagemaker
      # Only runs if quality-gate passed.
```

**The flow:**
1. Train the model. Evaluate it. Save metrics.
2. Run `pytest -m model`. If AUC < 0.90 or recall < 0.02, the tests FAIL.
3. If tests fail, the pipeline stops. No deployment. The team is notified.
4. If tests pass, deployment proceeds.

This pattern ensures that NO model is deployed unless it meets quality requirements.
It is an automated safety net that does not depend on humans remembering to check metrics.

---

## 6. Testing Best Practices for ML

### Test Data, Not Just Code

Traditional testing focuses on code behavior: "does this function return the right
value?" ML testing must also test data:

- **Schema tests:** Does the data have the expected columns and types?
- **Distribution tests:** Are feature distributions within expected ranges?
- **Null tests:** Are there unexpected nulls?
- **Uniqueness tests:** Are IDs truly unique?
- **Referential integrity:** Do foreign keys reference valid records?
- **Freshness tests:** Is the data recent enough?

Tools for data testing:
- **Great Expectations:** Define expectations about your data ("this column is never
  null," "this column's values are between 0 and 1") and validate automatically.
- **Pandera:** DataFrame validation library that integrates with pandas and pytest.
- **dbt tests:** If your data comes from a SQL warehouse, dbt can run data quality tests.

### Property-Based Testing for ML (Hypothesis Library)

Instead of testing with specific examples, property-based testing generates random inputs
and checks that properties hold:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(X=hnp.arrays(np.float64, (st.integers(1, 100), 10),
       elements=st.floats(-10, 10, allow_nan=False)))
def test_predictions_always_sum_to_one(X):
    model = load_model()
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)
```

**What this does:**
- Hypothesis generates hundreds of random input arrays.
- Each array has 1-100 rows and 10 columns.
- Values are floats between -10 and 10 (no NaN).
- The test checks that probabilities sum to 1.0 for ALL generated inputs.

**Why this is powerful:** It can find edge cases you never thought of:
- What if all features are exactly 0?
- What if one feature is -10 and all others are 10?
- What if there is only one sample?

### Snapshot Testing for Model Outputs

Snapshot testing captures a model's output on a fixed dataset and compares future outputs
against this snapshot. If the outputs change, the test fails.

```python
def test_model_predictions_match_snapshot():
    X_snapshot = np.load("tests/fixtures/snapshot_input.npy")
    expected = np.load("tests/fixtures/snapshot_predictions.npy")
    model = load_model("models/current_model.json")
    actual = model.predict_proba(X_snapshot)
    np.testing.assert_array_almost_equal(actual, expected, decimal=6)
```

**When to use snapshot testing:**
- After deploying a model, create a snapshot. If someone accidentally changes the model
  file or the preprocessing code, the snapshot test catches it.
- Not suitable during active development (snapshots need updating with every change).

### Testing with Production Data Samples

Keep a curated set of real production examples for testing:

```python
# tests/fixtures/production_samples.json
[
    {"features": {...}, "expected_class": 0, "description": "normal grocery purchase"},
    {"features": {...}, "expected_class": 1, "description": "known fraud pattern A"},
    {"features": {...}, "expected_class": 0, "description": "high amount but legitimate"},
    {"features": {...}, "expected_class": 1, "description": "known fraud pattern B"},
]
```

These "golden examples" serve as regression tests: if the model stops correctly
classifying known fraud patterns, something is wrong.

**Privacy consideration:** anonymize or synthesize the production samples. Never commit
real customer data to a test repository.

### Flaky Tests in ML and How to Handle Them

A flaky test is one that sometimes passes and sometimes fails without any code change.
ML tests are especially prone to flakiness because of:

1. **Floating-point non-determinism:** Different hardware (CPU vs GPU), different
   library versions, or different compilation flags can produce slightly different
   floating-point results. Fix: use tolerances (`np.allclose`, `abs(x - y) < epsilon`).

2. **Random seed propagation:** If a test relies on global random state, other tests
   can affect it. Fix: use isolated RNGs (`np.random.default_rng(seed)`), as this
   project does.

3. **Resource-dependent tests:** Tests that depend on external services (APIs, databases,
   cloud services) fail when those services are unavailable. Fix: mock external
   dependencies in unit tests. Only use real services in integration tests with retry
   logic.

4. **Threshold sensitivity:** A model quality test with `assert recall >= 0.80` might
   get 0.799 on some runs. Fix: set thresholds with a margin. If the real requirement
   is 0.80, test for 0.75 in automated tests and review the actual value in dashboards.

5. **Training non-determinism:** Even with the same seed, different hardware or library
   versions can produce slightly different models. Fix: for unit tests, use pre-trained
   dummy models (as this project does). For model quality tests, set generous-enough
   thresholds.

**The golden rule for ML test thresholds:** Your automated test threshold should be
lower than your actual quality requirement. The test catches catastrophic failures.
Dashboards and human review catch gradual degradation.

### Summary: What to Test at Each Level

```
+------------------+----------------------------+-----------+------------------+
| Level            | What to Test               | Speed     | When to Run      |
+------------------+----------------------------+-----------+------------------+
| Unit             | Data transforms, utils,    | ms-sec    | Every commit     |
|                  | model output properties    |           |                  |
+------------------+----------------------------+-----------+------------------+
| Integration      | Pipeline end-to-end,       | sec-min   | Every PR         |
|                  | component interactions     |           |                  |
+------------------+----------------------------+-----------+------------------+
| Model Quality    | Business metrics meet bars | sec       | After training   |
|                  | (requires trained model)   |           | (CI/CD gate)     |
+------------------+----------------------------+-----------+------------------+
| Data Quality     | Schema, distributions,     | sec-min   | Every data       |
|                  | freshness, completeness    |           | pipeline run     |
+------------------+----------------------------+-----------+------------------+
| A/B / Shadow     | Real-world impact,         | days-wks  | Before full      |
|                  | business metrics           |           | rollout          |
+------------------+----------------------------+-----------+------------------+
```

### Interview Tips for Testing Questions

**Q: "How do you test ML models?"**
A: "I use a testing pyramid approach. Unit tests verify data transformations and model
output properties (probabilities sum to 1, outputs are binary). Integration tests verify
the full pipeline produces correctly shaped data with expected distributions. Model
quality tests act as a CI/CD gate, checking that trained models meet minimum business
metrics (AUC, recall, precision). I also monitor production data quality and model
performance continuously."

**Q: "What is the most important test for a fraud detection model?"**
A: "The most important tests are: (1) the model is not degenerate, meaning it predicts
both fraud and non-fraud, not just one class; (2) recall meets the business threshold,
because a missed fraud case is more costly than a false alarm; and (3) the preprocessing
pipeline produces correctly scaled features, because scaling bugs can silently corrupt
all predictions."

**Q: "How do you handle flaky tests in ML?"**
A: "First, isolate randomness using per-test random number generators with fixed seeds
instead of global random state. Second, use approximate assertions with explicit
tolerances for floating-point comparisons. Third, set test thresholds conservatively
below the actual business requirement to absorb natural variance. Fourth, mock external
dependencies in unit tests and use retry logic in integration tests."

---

**Key files referenced in this guide:**
- `tests/unit/test_preprocess.py` -- preprocessing unit tests
- `tests/unit/test_model.py` -- model behavior unit tests
- `tests/integration/test_pipeline.py` -- pipeline integration tests
- `tests/model/test_model_quality.py` -- model quality gate tests
- `pyproject.toml` -- pytest configuration (markers, testpaths, addopts)
