# Guide 13: Monitoring Deep Dive -- Drift Detection & Performance Tracking

## Table of Contents
1. [Why ML Models Fail in Production -- Real Examples](#1-why-ml-models-fail-in-production)
2. [DriftDetector Class -- Every Method, Every Line](#2-driftdetector-class)
3. [run_drift_detection() -- The Orchestrator](#3-run_drift_detection)
4. [PerformanceMonitor Class -- Every Method](#4-performancemonitor-class)
5. [Monitoring Strategy for Production ML](#5-monitoring-strategy-for-production-ml)

---

## 1. Why ML Models Fail in Production

### The Fundamental Problem

Traditional software is deterministic. If you deploy a function that adds two numbers, it
will keep adding two numbers correctly forever. ML models are different. They learn patterns
from historical data, and when the world changes, those patterns become stale or wrong.

A model is a frozen snapshot of past relationships. The world does not stay frozen.

### Data Drift: Feature Distributions Change

**What it is:** The statistical distribution of input features changes between training
time and inference time. The model was trained on data that looked one way, and now the
incoming data looks different.

**Real-world example -- COVID and credit card fraud:**
Before March 2020, a fraud detection model learned that "transaction at 3 AM from a
foreign country for $5,000" was suspicious. Then COVID hit. People were buying online
at all hours, international e-commerce patterns changed, and average transaction amounts
shifted because people stopped buying coffee at cafes and started buying home office
equipment. The model's features (time-of-day distributions, amount distributions,
merchant category distributions) all shifted simultaneously. A model trained on
pre-COVID data would generate massive numbers of false positives or miss new fraud
patterns.

**Technical definition:** If P_train(X) is the distribution of features at training
time, and P_prod(X) is the distribution at production time, data drift means
P_train(X) != P_prod(X).

### Concept Drift: The Relationship Between Features and Target Changes

**What it is:** Even if the input distributions stay the same, the mapping from inputs
to outputs changes. The "concept" the model learned is no longer valid.

**Real-world example -- spam detection:**
In 2010, Nigerian prince emails were spam. The features (certain keywords, sender
patterns) mapped to "spam." By 2023, spammers adapted. They use AI-generated text that
looks legitimate. The features look different for the same concept (spam), and what
used to be a strong signal (broken English) is no longer relevant. The concept of
"what makes an email spam" drifted.

**Technical definition:** P(Y | X) changes over time. The conditional distribution of
the target given the features is different now than when the model was trained.

### Upstream Data Issues: Pipeline Breaks

**What it is:** A data pipeline feeding your model breaks, sending nulls, wrong values,
stale data, or data in an unexpected format.

**Real-world example -- feature store outage:**
A major bank had a fraud model that used a feature "average_transaction_last_30_days."
The feature store had an outage and started returning 0.0 for every customer instead of
the real average. The model, seeing that every customer had an average of $0, started
flagging every transaction as suspicious because the sudden drop looked anomalous. It
took 4 hours to notice, during which thousands of legitimate transactions were blocked.

### Schema Changes: Structural Data Changes

**What it is:** A column is added, removed, renamed, or its data type changes in the
upstream data source.

**Real-world example:**
An engineering team renamed a column from `user_age` to `customer_age` in the database
without informing the ML team. The model's preprocessing code expected `user_age`,
threw a KeyError, and the prediction service returned 500 errors for every request.
This is why schema validation (like Great Expectations or Pandera) is critical.

### Zillow's $500M Loss -- A Cautionary Tale

Zillow's iBuying program (Zillow Offers) used ML models to predict home prices and make
instant purchase offers. In 2021, the models failed catastrophically:

1. **Data drift:** Post-COVID housing market was unlike anything in the training data.
   Bidding wars, remote work migration, and low interest rates created unprecedented
   price movements.
2. **Concept drift:** The historical relationship between features (square footage,
   location, comparable sales) and price changed. Homes that "should" cost $400K based
   on historical patterns were selling for $600K.
3. **Feedback loops:** Zillow's own buying was affecting the market, pushing prices up
   in neighborhoods where they bought aggressively.
4. **Model overconfidence:** The models did not adequately communicate uncertainty.

Zillow ended up holding thousands of homes it had overpaid for. They wrote down $569
million and laid off 25% of their workforce. The CEO said, "We've determined the
unpredictability in forecasting home prices far exceeds what we anticipated."

**Lesson:** Monitoring is not optional. If Zillow had robust drift detection that
triggered retraining or halted buying when the model was uncertain, the damage could
have been limited.

---

## 2. DriftDetector Class -- Every Method, Every Line

**Source file:** `src/monitoring/drift_detection.py`

### Imports and Setup

```python
import json
from datetime import datetime
from typing import Any

import boto3
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import get_aws_config, get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

**Line-by-line:**

- `json` -- for serializing drift reports to JSON files. JSON is the standard format
  for machine-readable reports that other tools (dashboards, alerting systems) can consume.

- `datetime` -- for timestamping drift reports. Every monitoring event needs a timestamp
  so you can correlate drift with deployments, incidents, or data pipeline changes.

- `typing.Any` -- type hint for the flexible dictionary structures in drift results.
  The `Any` type is used because drift results contain a mix of floats, booleans,
  strings, and nested dicts.

- `boto3` -- AWS SDK. Used to publish drift metrics to CloudWatch for alerting. In
  production, when drift is detected, you want an automated alarm, not just a log line.

- `numpy` -- numerical operations. Used for percentile calculations, histograms, and
  clipping operations in PSI calculation.

- `pandas` -- data manipulation. Both reference data and current data come as DataFrames.

- `scipy.stats` -- statistical tests. Specifically `ks_2samp` for the Kolmogorov-Smirnov
  two-sample test, the primary statistical test used for drift detection.

- `get_aws_config, get_project_root, load_params` -- project utilities for configuration
  management. Centralizing config avoids hardcoded paths and region names.

- `get_logger(__name__)` -- creates a logger named after this module
  (`src.monitoring.drift_detection`). This makes log filtering easy: you can grep for
  drift-related logs without noise from other modules.

### The `__init__` Method

```python
class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)
```

**What is reference data?**
The reference data is a snapshot of what your model was trained on. Think of it as the
"known good" distribution. When new data arrives in production, you compare it against
this reference to see if something changed.

**`reference_data: pd.DataFrame`** -- the full training dataset (or a representative
sample of it). You store the entire DataFrame, not just summary statistics, because the
KS test needs access to every individual data point to compute the empirical CDF.

**`threshold: float = 0.05`** -- the p-value cutoff for the KS test. If the p-value
is below this threshold, we reject the null hypothesis that the two distributions are
the same. The default 0.05 means "5% significance level," the standard in statistics.

**Why 0.05?** This is a convention from frequentist statistics. A p-value of 0.05 means
there is a 5% chance of falsely declaring drift when there is none (Type I error / false
positive). In production, you might adjust this:
- Lower (0.01) if false drift alarms are expensive (alert fatigue).
- Higher (0.10) if missing real drift is expensive (fraud goes undetected).

**`self.reference_stats = self._compute_stats(reference_data)`** -- precomputes summary
statistics at initialization time. This is the "eager computation" pattern: compute once,
use many times. The stats are stored for potential use in reporting or quick comparisons
without needing the full KS test.

### The `_compute_stats` Method

```python
def _compute_stats(self, df: pd.DataFrame) -> dict:
    return {
        col: {"mean": df[col].mean(), "std": df[col].std(), "median": df[col].median()}
        for col in df.select_dtypes(include=[np.number]).columns
    }
```

This is a dictionary comprehension that builds a nested dictionary of statistics for
every numeric column.

**`df.select_dtypes(include=[np.number])`** -- filters to only numeric columns. This is
critical because:
1. You cannot compute mean/std/median on string or categorical columns.
2. In the fraud dataset, all V1-V28 features are numeric (PCA-transformed), plus Amount.
3. `np.number` matches all numeric dtypes: int8, int16, int32, int64, float16, float32,
   float64, etc. Using `np.number` rather than listing specific types is more robust.

**`.columns`** -- returns the Index of column names that survived the dtype filter.

**Why mean, std, and median?**
- `mean` -- the central tendency. If the mean of a feature shifts, the "average" input
  to the model has changed.
- `std` -- the spread/variability. If std increases, the data is more dispersed; if it
  decreases, the data is more concentrated. Both can affect model behavior.
- `median` -- the robust central tendency. Unlike the mean, the median is resistant to
  outliers. If the mean shifts but the median does not, the shift is driven by extreme
  values, not a general distributional change.

**The resulting structure looks like:**
```python
{
    "V1": {"mean": 0.0012, "std": 1.95, "median": 0.018},
    "V2": {"mean": 0.0008, "std": 1.65, "median": -0.005},
    "Amount": {"mean": 88.34, "std": 250.12, "median": 22.0},
    ...
}
```

### The `detect_drift` Method -- Line by Line

This is the core of the drift detection system. Let us go through every line.

```python
def detect_drift(self, current_data: pd.DataFrame) -> dict[str, Any]:
```

**Method signature:**
- Takes `current_data` -- the new/production data to compare against the reference.
- Returns `dict[str, Any]` -- a structured report with drift results for every feature.
- The `dict[str, Any]` syntax (Python 3.9+) is equivalent to `Dict[str, Any]` from
  `typing`. The `Any` type is used because values include strings, floats, bools, and
  nested dicts.

```python
    features: dict[str, Any] = {}
    results: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "drifted": False,
    }
```

**`features: dict[str, Any] = {}`** -- initializes an empty dict that will hold per-feature
drift results. The type annotation `dict[str, Any]` is a hint for developers and type
checkers; it does not affect runtime behavior.

**`datetime.utcnow().isoformat()`** -- generates an ISO 8601 timestamp like
`"2024-03-15T14:32:07.123456"`. Key details:
- `utcnow()` returns the current time in UTC, not local time. Always use UTC for
  monitoring timestamps. If you use local time and your servers are in different time
  zones, correlating events becomes a nightmare.
- `.isoformat()` formats as ISO 8601, the international standard for datetime strings.
  This format is sortable, parseable by every language, and unambiguous.

**`"features": features`** -- note that `features` is a reference to the same dict
object created on the line above. When we modify `features` later in the loop, those
changes appear in `results["features"]` automatically. This is Python's reference
semantics for mutable objects.

**`"drifted": False`** -- starts as False (optimistic assumption: no drift). Gets flipped
to True if any individual feature is found to be drifted.

```python
    for col in self.reference.select_dtypes(include=[np.number]).columns:
        if col not in current_data.columns:
            continue
```

**Iterating over reference columns:** We iterate over the reference data's numeric columns,
not the current data's columns. This is deliberate:
- If the current data is missing a column that the reference had, we skip it (the
  `continue` branch). In a more robust system, a missing column would itself be an alert.
- If the current data has extra columns that the reference did not have, we ignore them.
  We only check features the model was trained on.

**`if col not in current_data.columns: continue`** -- defensive programming. In
production, data schemas can change. A column might be dropped by an upstream pipeline
change. Instead of crashing with a KeyError, we skip and continue. In a production
system, you would also log a warning here.

```python
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference[col], current_data[col])
```

**The Kolmogorov-Smirnov Two-Sample Test**

This is the primary statistical test for drift detection. Let us unpack it thoroughly.

**What is the KS test?**
The KS test compares two samples and asks: "Could these two samples have been drawn from
the same underlying distribution?" It is a non-parametric test, meaning it makes no
assumptions about what that underlying distribution is (normal, exponential, etc.).

**How it works -- the intuition:**
1. For each sample, compute the empirical cumulative distribution function (ECDF).
   The ECDF at value x is the proportion of sample values less than or equal to x.
2. The KS statistic is the maximum absolute difference between the two ECDFs at any
   point. Visually, if you plot both ECDFs, the KS statistic is the tallest vertical
   gap between the two curves.

```
    1.0 |        ___________
        |       /    ___---
        |      / ___/
        |     //    <-- KS statistic = max gap here
        |    /|
        |   / |
    0.0 |__/  |
        +-----|------------>
              x
```

3. The p-value tells you the probability of seeing a KS statistic this large (or larger)
   if the two samples really did come from the same distribution.

**Return values:**
- `ks_stat` (float, 0 to 1): The maximum distance. 0 means identical distributions.
  1 means completely non-overlapping.
- `ks_pvalue` (float, 0 to 1): Small p-value means the distributions are likely
  different. Large p-value means we cannot distinguish them.

**Why KS test for drift detection?**
1. Non-parametric: works on any distribution shape. Credit card features (especially
   PCA-transformed ones) can have weird distributions.
2. Sensitive to any type of change: shifts in location, spread, or shape.
3. Well-understood, with known statistical properties.
4. Fast to compute: O(n log n) where n is the sample size.

**Limitations:**
- Less sensitive in the tails of distributions (where fraud often lives).
- Sensitive to sample size: with very large samples, even tiny, practically meaningless
  differences become "statistically significant."

```python
        psi = self._calculate_psi(self.reference[col], current_data[col])
```

This calls the PSI calculation (covered in detail below). PSI gives a complementary
view of drift that is more commonly used in banking and finance.

```python
        is_drifted = bool(ks_pvalue < self.threshold)
```

**The drift decision:** If the KS p-value is below the threshold (default 0.05), we
declare this feature as drifted. The `bool()` wrapper ensures we get a Python bool
rather than a numpy bool_ type. This matters for JSON serialization -- `json.dump`
handles Python bools natively but might choke on numpy bools.

**Why base the decision on p-value, not KS statistic?**
The KS statistic alone does not tell you if the difference is "significant." A KS
statistic of 0.05 might be significant with 100,000 samples but not with 100 samples.
The p-value accounts for sample size.

```python
        features[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": round(float(psi), 6),
            "drifted": is_drifted,
            "ref_mean": round(float(self.reference[col].mean()), 6),
            "cur_mean": round(float(current_data[col].mean()), 6),
        }
```

**Building the per-feature report:**
- `round(float(...), 6)` -- two conversions happening:
  - `float()` converts from numpy float64 to Python native float (JSON serializable).
  - `round(..., 6)` limits to 6 decimal places for readability. Nobody needs 15 decimal
    places in a monitoring report.
- `ks_statistic` and `ks_pvalue` -- the raw KS test results.
- `psi` -- the Population Stability Index result.
- `drifted` -- the boolean decision.
- `ref_mean` and `cur_mean` -- the mean of the feature in reference vs current data.
  This provides immediate interpretability: "V14 mean went from 0.003 to 0.872."

```python
        if is_drifted:
            results["drifted"] = True
```

**Global drift flag:** If any single feature is drifted, the entire result is marked
as drifted. This is a conservative approach: one bad feature is enough to raise the
alarm. In practice, you might want a more nuanced approach, like "drift only if more
than 20% of features are drifted" -- which is what `drift_ratio` enables.

```python
    n_drifted = sum(1 for f in features.values() if f["drifted"])
    results["n_features_drifted"] = n_drifted
    results["n_features_total"] = len(features)
    results["drift_ratio"] = round(n_drifted / max(len(features), 1), 4)
```

**Summary statistics:**

- `sum(1 for f in features.values() if f["drifted"])` -- a generator expression that
  counts how many features were flagged as drifted. This is Pythonic counting: generate
  a 1 for each drifted feature, then sum them.

- `n_features_drifted` -- the count (e.g., 3 out of 29 features drifted).

- `n_features_total` -- total features checked.

- `drift_ratio` -- the proportion of drifted features. `max(len(features), 1)` prevents
  division by zero if there are no numeric features. The ratio is useful for nuanced
  alerting: "page the on-call if drift_ratio > 0.5, send a Slack message if > 0.1."

### The `_calculate_psi` Method -- Line by Line

```python
def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index -- measures distribution shift."""
```

**What is PSI?**
PSI (Population Stability Index) was developed in the credit scoring industry to
measure how much a population (the applicant pool) has shifted over time. It is
standard practice in banking regulators' model risk management frameworks.

**Key difference from KS test:** The KS test gives a p-value (statistical significance).
PSI gives a magnitude (how much shift). They answer different questions:
- KS: "Is there a statistically significant difference?" (yes/no with a probability)
- PSI: "How big is the shift?" (a continuous score with interpretive ranges)

```python
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
```

**Creating percentile-based bins from the reference distribution.**

- `np.linspace(0, 100, bins + 1)` -- generates evenly spaced numbers from 0 to 100.
  With `bins=10`, this produces `[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]` --
  eleven values that define ten bins.

- `np.percentile(reference, ...)` -- computes the percentile values of the reference
  data at those points. For example, if the reference data for "Amount" ranges from
  $0 to $25,000, the 10th percentile might be $1.50, the 20th percentile $5.00, etc.

**Why percentile-based bins instead of equal-width bins?**
Equal-width bins (e.g., $0-$2500, $2500-$5000, ...) would put 95% of credit card
transactions into the first bin because transaction amounts are heavily right-skewed.
Percentile-based bins ensure each bin has roughly the same number of reference
observations, giving equal "weight" to each part of the distribution.

```python
    breakpoints = np.unique(breakpoints)
```

**Handling duplicate breakpoints.** If many values in the reference are identical (e.g.,
a feature where 30% of values are 0.0), multiple percentiles will map to the same value.
`np.unique` removes duplicates. Without this, `np.histogram` would error on bins with
zero width. This is a subtle but critical defensive step.

```python
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)
```

**Binning and converting to proportions.**

- `np.histogram(reference, bins=breakpoints)` -- counts how many reference values fall
  into each bin. Returns a tuple: `(counts_array, bin_edges_array)`. The `[0]` extracts
  just the counts.

- `/ len(reference)` -- converts raw counts to proportions. If 100 out of 1000
  reference values fall in bin 3, the proportion is 0.10. We need proportions (not
  counts) because PSI compares relative distributions, making it independent of
  sample sizes.

- The same is done for `current` data using the same breakpoints. This is crucial:
  both histograms must use the same bin edges (defined by the reference) so we are
  comparing equivalent regions of the feature space.

```python
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
```

**Clipping to avoid mathematical catastrophe.**

- `np.clip(ref_counts, 1e-6, None)` -- replaces any value below 1e-6 (0.000001)
  with 1e-6. The upper bound `None` means no upper clipping.

**Why is this necessary?** The PSI formula involves:
1. Division: `cur_counts / ref_counts` -- if `ref_counts` is 0, we get division by zero.
2. Logarithm: `log(cur_counts / ref_counts)` -- if the ratio is 0, we get log(0) = -inf.

By clipping to 1e-6, we ensure:
- No division by zero.
- No log(0).
- The tiny value (0.0001%) is small enough to not materially affect the PSI calculation.

This is a standard numerical stability technique used throughout scientific computing.

```python
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
```

**The PSI formula itself:**

PSI = sum over all bins of: (current% - reference%) * ln(current% / reference%)

Breaking it down:
- `cur_counts - ref_counts` -- the difference in proportions. If bin 3 has 15% of
  current data but only 10% of reference data, the difference is +0.05.
- `np.log(cur_counts / ref_counts)` -- the log ratio. If current has 15% and reference
  has 10%, the log ratio is ln(1.5) = 0.405.
- Multiplying them: 0.05 * 0.405 = 0.020. This bin contributes 0.020 to the total PSI.
- `np.sum(...)` -- sums contributions from all bins.

**Why this specific formula?**
PSI is derived from the Kullback-Leibler (KL) divergence. It is actually a symmetric
version: PSI = KL(current || reference) + KL(reference || current). This symmetry is
important because it means PSI(A, B) == PSI(B, A), unlike raw KL divergence.

**PSI interpretation ranges (industry standard):**

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant drift | Continue monitoring |
| 0.10 - 0.20 | Moderate drift | Investigate, consider retraining |
| > 0.20 | Significant drift | Retrain model, investigate root cause |
| > 0.25 | Severe drift | Potentially halt predictions, urgent investigation |

**Real-world example -- PSI detecting seasonal drift:**
A credit card fraud model has a feature "transaction_amount." During November-December
(holiday shopping season), the distribution of transaction amounts shifts right (higher
amounts). PSI might go from 0.05 in September to 0.22 in December. This tells the
team: "The amount distribution has shifted significantly. The model may not perform
well on holiday spending patterns. Consider retraining with recent data or using a
season-specific model."

---

## 3. run_drift_detection() -- The Orchestrator

```python
def run_drift_detection():
    params = load_params()
    root = get_project_root()
    processed_dir = root / params["data"]["processed_path"]
```

**Loading configuration and paths:**
- `load_params()` -- reads `params.yaml` which contains all hyperparameters and
  configuration, including `monitoring.drift_threshold`.
- `get_project_root()` -- returns the project root directory as a `Path` object.
- `processed_dir` -- the directory containing processed data files (e.g.,
  `data/processed/`).

```python
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
```

**Loading reference and current data.**

- `X_train` serves as the **reference data** -- the distribution the model was trained on.
- `X_test` serves as the **current data** -- a proxy for production data.

**In a real production system, the data sources would be different:**
- Reference: A snapshot of the training data stored in S3, a feature store, or a
  database at training time. This never changes until you retrain.
- Current: The last N hours/days of production inference data. This is collected from
  the prediction service's request logs or a streaming pipeline.

**Why use X_test as a proxy?** In this learning project, we do not have a live
production system. X_test simulates "new data" that was not used in training. In
a real system, you would replace this with a query to your production data warehouse.

```python
    detector = DriftDetector(
        reference_data=X_train,
        threshold=params["monitoring"]["drift_threshold"],
    )
    results = detector.detect_drift(X_test)
```

**Creating the detector and running detection:**
- The threshold comes from `params.yaml`, not hardcoded. This follows the configuration
  externalization pattern: you can change the threshold without modifying code.
- `detect_drift(X_test)` returns the full drift report.

```python
    reports_dir = root / "metrics" / "drift"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
```

**Saving the drift report with a timestamped filename.**

- `reports_dir.mkdir(parents=True, exist_ok=True)` -- creates the directory and all
  parent directories if they do not exist. `exist_ok=True` means no error if the
  directory already exists. This is idempotent -- safe to run multiple times.

- `datetime.utcnow().strftime('%Y%m%d_%H%M%S')` -- formats the current UTC time as
  `20240315_143207`. This creates unique, sortable filenames like
  `drift_report_20240315_143207.json`.

- **The audit trail pattern:** Every drift check produces a new file rather than
  overwriting the previous one. This creates a historical record. You can:
  - See how drift evolved over time.
  - Correlate drift events with deployments, incidents, or data pipeline changes.
  - Satisfy regulatory requirements (in banking, you must prove you monitored your
    models).

- `json.dump(results, f, indent=2)` -- writes the JSON with 2-space indentation for
  human readability. In a high-throughput system, you might omit `indent` to save
  disk space and write speed.

```python
    if results["drifted"]:
        logger.warning(
            f"DRIFT DETECTED: {results['n_features_drifted']}"
            f"/{results['n_features_total']} features"
        )
        _send_drift_alarm(results, params)
    else:
        logger.info("No drift detected")
```

**Conditional alerting:**
- If drift is detected, log a WARNING (not INFO). This is important because log
  aggregation tools (Splunk, CloudWatch Logs, Datadog) can filter by severity level.
  WARNING logs trigger different alert rules than INFO logs.
- `_send_drift_alarm` publishes a metric to CloudWatch, which can trigger alarms
  (email, PagerDuty, Slack).
- If no drift, just log INFO. No alarm needed.

```python
    logger.info(f"Drift report saved to {report_path}")
    return results
```

Always log where the report was saved so an operator can find it.

### `_send_drift_alarm` -- CloudWatch Integration

```python
def _send_drift_alarm(results: dict, params: dict):
    try:
        aws_config = get_aws_config(params)
        cloudwatch = boto3.client("cloudwatch", region_name=aws_config["region"])
```

**Creating a CloudWatch client:**
- `get_aws_config(params)` -- extracts AWS configuration (region, credentials profile)
  from the params. Centralizing this avoids hardcoding `us-east-1` everywhere.
- `boto3.client("cloudwatch", ...)` -- creates a CloudWatch client. This is the service
  for publishing custom metrics and setting up alarms on AWS.

```python
        cloudwatch.put_metric_data(
            Namespace="MLOps/FraudDetection",
            MetricData=[
                {
                    "MetricName": "DataDriftDetected",
                    "Value": 1,
                    "Unit": "Count",
                },
                {
                    "MetricName": "DriftedFeatureRatio",
                    "Value": results["drift_ratio"],
                    "Unit": "None",
                },
            ],
        )
```

**Publishing custom metrics:**

- `Namespace="MLOps/FraudDetection"` -- organizes metrics hierarchically. All metrics
  from this fraud detection project are grouped under this namespace. In the CloudWatch
  console, you can browse by namespace.

- Two metrics are published:
  1. `DataDriftDetected` with value 1 -- a binary event. You can set a CloudWatch alarm:
     "If DataDriftDetected Sum > 0 in any 5-minute period, send a PagerDuty alert."
  2. `DriftedFeatureRatio` -- a continuous value (0.0 to 1.0). You can set a threshold
     alarm: "If DriftedFeatureRatio > 0.5, it means more than half the features drifted,
     escalate to senior engineer."

- `"Unit": "None"` -- CloudWatch requires a unit. "None" means the metric is
  dimensionless (a ratio).

**Real-world example -- setting up the alarm chain:**
```
Drift detected
  -> CloudWatch metric published
    -> CloudWatch alarm triggers (threshold: drift_ratio > 0.3)
      -> SNS topic notified
        -> Lambda function triggered
          -> Sends Slack message: "@ml-team: Drift detected in fraud model.
             7/29 features drifted (24%). Review drift report at s3://..."
          -> If drift_ratio > 0.5, also pages on-call engineer
```

```python
    except Exception as e:
        logger.warning(f"Failed to publish drift alarm: {e}")
```

**Graceful failure:** If CloudWatch publishing fails (network issue, wrong credentials,
AWS service outage), we log a warning and continue. We do not crash. The drift report
was already saved to disk. The monitoring system's failure should not prevent the drift
detection result from being recorded.

This is a critical design pattern for monitoring code: **monitoring should never break
the system it monitors.**

---

## 4. PerformanceMonitor Class -- Every Method

**Source file:** `src/monitoring/performance.py`

### Imports

```python
import json
from datetime import datetime

import boto3
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.utils.config import get_aws_config, get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

The imports follow the same pattern as drift detection. The key addition is
`sklearn.metrics` which provides all the classification metrics we need.

### The `__init__` Method

```python
class PerformanceMonitor:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
```

Simple initialization. `thresholds` is a dictionary like:
```python
{
    "min_recall": 0.02,
    "min_precision": 0.50,
    "min_f1": 0.04,
    "min_auc_roc": 0.90
}
```

These are the minimum acceptable values for each metric. If any metric falls below
its threshold, the model is flagged as degraded.

**Why thresholds and not just metrics?**
Metrics alone tell you "precision is 0.45." Thresholds tell you "precision is 0.45,
which is below the minimum of 0.50 -- this is a problem." Thresholds encode business
requirements into code.

### The `evaluate` Method -- Line by Line

```python
def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
```

**Three inputs:**
- `y_true` -- ground truth labels (0 or 1). In production, these come from delayed
  feedback: the bank eventually confirms whether a transaction was actually fraudulent.
- `y_pred` -- binary predictions (0 or 1). What the model predicted.
- `y_prob` -- probability scores (0.0 to 1.0). The model's confidence. Needed for
  AUC-ROC, which evaluates the model across all possible thresholds.

```python
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "prediction_positive_rate": float(y_pred.mean()),
    }
```

**Each metric explained:**

- `precision_score(y_true, y_pred, zero_division=0)` -- of all transactions the model
  flagged as fraud, what fraction actually were fraud? `zero_division=0` means: if the
  model predicted zero positives (never flagged anything), return 0 instead of raising
  a warning. This handles the degenerate case gracefully.

- `recall_score(y_true, y_pred, zero_division=0)` -- of all actual fraud transactions,
  what fraction did the model catch? This is the most business-critical metric for fraud
  detection. Missing a $10,000 fraud (low recall) is much worse than investigating a
  $50 legitimate transaction (low precision).

- `f1_score(y_true, y_pred, zero_division=0)` -- the harmonic mean of precision and
  recall. Balances both concerns. A model with 100% recall and 0.1% precision (flags
  everything) would have a terrible F1 score.

- `roc_auc_score(y_true, y_prob)` -- Area Under the Receiver Operating Characteristic
  curve. Measures the model's ability to rank fraud higher than non-fraud, across all
  possible decision thresholds. AUC of 0.5 means random guessing; 1.0 means perfect
  separation.

- `n_samples` -- how many samples were evaluated. Important for context: metrics on
  100 samples are much noisier than on 100,000 samples.

- `n_positive` -- how many actual fraud cases. `y_true.sum()` works because fraud is
  coded as 1 and legitimate as 0, so summing gives the count of 1s.

- `prediction_positive_rate` -- `y_pred.mean()` gives the fraction of predictions that
  are positive (fraud). If the model suddenly predicts 50% fraud when the base rate is
  0.17%, something is very wrong. This is a quick sanity check.

```python
    alerts = []
    if metrics["recall"] < self.thresholds["min_recall"]:
        alerts.append(f"recall {metrics['recall']:.4f} < {self.thresholds['min_recall']}")
    if metrics["precision"] < self.thresholds["min_precision"]:
        min_prec = self.thresholds["min_precision"]
        alerts.append(f"precision {metrics['precision']:.4f} < {min_prec}")
    if metrics["f1"] < self.thresholds["min_f1"]:
        alerts.append(f"f1 {metrics['f1']:.4f} < {self.thresholds['min_f1']}")
```

**Alert generation:**

- `alerts = []` -- starts with an empty list. Each threshold violation adds a
  human-readable alert message.

- Each `if` statement checks one metric against its threshold. The alert message
  includes both the actual value and the threshold, making it immediately actionable:
  `"recall 0.0180 < 0.02"` tells you exactly what failed and by how much.

- `:.4f` -- formats to 4 decimal places. Enough precision for metrics like precision
  and recall without being overwhelming.

- Note: `auc_roc` is checked in the `evaluate` method's thresholds but not in the
  alerts list here. The `run_performance_monitoring` function handles AUC-ROC
  separately.

```python
    metrics["alerts"] = alerts
    metrics["degraded"] = len(alerts) > 0
    return metrics
```

**`metrics["degraded"] = len(alerts) > 0`** -- a simple boolean pattern. If there are
any alerts (the list is non-empty), the model is degraded. `len(alerts) > 0` evaluates
to `True` if any threshold was violated, `False` otherwise.

This is cleaner than maintaining a separate boolean variable and flipping it in each
`if` block. It derives the degradation status from the alerts list -- single source
of truth.

### `run_performance_monitoring()` -- The Orchestrator

```python
def run_performance_monitoring():
    params = load_params()
    root = get_project_root()

    eval_metrics_path = root / "metrics" / "eval_metrics.json"
    if not eval_metrics_path.exists():
        logger.warning("No evaluation metrics found -- run evaluate first")
        return
```

**Graceful handling of missing data:**
The function first checks if evaluation metrics exist. If they do not (because the
model has not been evaluated yet), it logs a warning and returns `None`. It does not
crash. This is important because monitoring might be scheduled to run on a cron job
regardless of whether evaluation has completed.

```python
    with open(eval_metrics_path) as f:
        eval_metrics = json.load(f)
```

**Loading persisted metrics.** The evaluation pipeline (a separate step) saves metrics
to `eval_metrics.json`. The monitoring pipeline reads them. This loose coupling means
evaluation and monitoring can run independently on different schedules.

```python
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "eval_metrics": eval_metrics,
        "thresholds": params["thresholds"],
        "degraded": False,
        "alerts": [],
    }
```

**Initializing the performance report.** Note that the report includes both the metrics
AND the thresholds. This is good practice for reproducibility: anyone reading the report
can see both what happened and what the expectations were.

```python
    for metric_name in ["recall", "precision", "f1", "auc_roc"]:
        threshold_key = f"min_{metric_name}"
        if eval_metrics.get(metric_name, 1.0) < params["thresholds"][threshold_key]:
            report["degraded"] = True
            report["alerts"].append(
                f"{metric_name}: {eval_metrics[metric_name]:.4f}"
                f" < {params['thresholds'][threshold_key]}"
            )
```

**The threshold checking loop -- dynamic metric iteration:**

- `for metric_name in ["recall", "precision", "f1", "auc_roc"]` -- iterates over the
  four metrics to check. This is more maintainable than four separate `if` blocks.
  Adding a new metric requires only adding a string to this list.

- `threshold_key = f"min_{metric_name}"` -- constructs the threshold key dynamically.
  `"recall"` becomes `"min_recall"`, `"auc_roc"` becomes `"min_auc_roc"`. This naming
  convention enables the dynamic loop.

- `eval_metrics.get(metric_name, 1.0)` -- safely gets the metric value, defaulting to
  1.0 if the key does not exist. The default of 1.0 is deliberately high: if a metric
  is missing, do not trigger an alert (1.0 is above any reasonable threshold). This is
  the "fail open" approach -- missing data does not trigger false alarms.

- `params["thresholds"][threshold_key]` -- gets the threshold from `params.yaml`.

- When a violation is found:
  - `report["degraded"] = True` -- set once, stays True. Even if later metrics are fine,
    one violation is enough.
  - The alert string includes the metric name, actual value, and threshold.

```python
    reports_dir = root / "metrics" / "performance"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"perf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
```

**Saving the performance report.** Same pattern as drift reports: timestamped filenames,
create directory if needed, JSON format. These reports accumulate over time, creating a
performance history.

```python
    if report["degraded"]:
        logger.warning(f"MODEL DEGRADED: {report['alerts']}")
        _publish_degradation_alarm(report, params)
    else:
        logger.info("Model performance is within acceptable thresholds")

    return report
```

**Conditional alerting.** Same pattern as drift: WARNING for problems, INFO for all-clear,
CloudWatch alarm for automated response.

### `_publish_degradation_alarm` -- CloudWatch Integration

```python
def _publish_degradation_alarm(report: dict, params: dict):
    try:
        aws_config = get_aws_config(params)
        cloudwatch = boto3.client("cloudwatch", region_name=aws_config["region"])
        cloudwatch.put_metric_data(
            Namespace="MLOps/FraudDetection",
            MetricData=[
                {
                    "MetricName": "ModelDegraded",
                    "Value": 1,
                    "Unit": "Count",
                },
            ],
        )
        logger.info("Degradation alarm published to CloudWatch")
    except Exception as e:
        logger.warning(f"Failed to publish degradation alarm: {e}")
```

**Same structure as `_send_drift_alarm`** but publishes a different metric:
`ModelDegraded`. This allows separate CloudWatch alarms for drift vs degradation:
- Drift alarm: "The input data has changed." Action: investigate data pipeline.
- Degradation alarm: "The model's predictions are bad." Action: investigate model,
  potentially rollback to previous version.

**Real-world example -- precision drops from 0.95 to 0.60:**
A data pipeline change accidentally stopped populating a feature (it returned 0 for
all transactions). The model relied on this feature heavily. Precision dropped from
0.95 to 0.60, meaning 40% of "fraud" alerts were false positives.

The monitoring system:
1. `run_performance_monitoring()` detects precision < min_precision threshold.
2. `_publish_degradation_alarm` fires.
3. CloudWatch alarm triggers PagerDuty.
4. On-call engineer gets paged at 2 AM.
5. Engineer checks the drift report (drift detection also fires, showing the broken
   feature has PSI > 1.0).
6. Engineer identifies the pipeline bug, fixes it, and verifies precision recovers.
7. Post-incident review: add a data quality check upstream to prevent recurrence.

---

## 5. Monitoring Strategy for Production ML

### What to Monitor -- The Three Pillars

**Pillar 1: Input Data Monitoring**
- Feature distributions (drift detection, as implemented above)
- Data quality: null rates, cardinality, value ranges
- Schema: column names, data types, expected columns present
- Volume: are we getting the expected number of predictions per hour?

**Pillar 2: Prediction Monitoring**
- Prediction distribution: is the model predicting the same ratio of fraud?
- Confidence distribution: are average confidence scores changing?
- Prediction latency: is the model slower than usual?
- Error rates: are prediction requests failing?

**Pillar 3: Ground Truth Monitoring (when available)**
- Actual performance metrics (precision, recall, F1, AUC)
- Performance per segment: is the model degraded for a specific customer group?
- Feedback delay: how long until you get ground truth labels?

For fraud detection, ground truth comes from:
- Chargebacks (customer disputes) -- delayed by 30-90 days
- Internal investigations -- delayed by 1-7 days
- Customer reports -- variable delay

This delay means you cannot rely solely on ground truth monitoring. You need data
drift detection as an early warning system.

### Monitoring Frequency

**Real-time (per-request):**
- Input schema validation (reject malformed requests immediately)
- Prediction latency
- Error rates
- Prediction confidence bounds (alert if confidence is extremely low)

**Hourly:**
- Input data distributions (mini-batch drift detection)
- Prediction distribution
- Request volume anomalies

**Daily:**
- Full drift report with statistical tests (KS, PSI)
- Performance metrics (if ground truth labels are available)
- Feature importance stability
- Model version tracking

**Weekly/Monthly:**
- Comprehensive model performance review
- Drift trend analysis (is drift accumulating slowly?)
- Retraining assessment
- Comparison against baseline/champion model

### Alert Fatigue -- Setting Thresholds That Do Not Cry Wolf

Alert fatigue is the number one killer of monitoring systems. If your team gets 50
drift alerts per day, they start ignoring all of them -- including the real ones.

**Strategies:**

1. **Tiered alerting:**
   - INFO: PSI 0.10-0.15 for any feature. Logged, not alerted.
   - WARNING: PSI > 0.20 for more than 3 features, or any feature PSI > 0.30.
     Slack message to the team channel.
   - CRITICAL: Performance metric below threshold. Pages the on-call engineer.

2. **Aggregation windows:** Do not alert on every 5-minute window. Aggregate over an
   hour. If drift is detected consistently for 3 consecutive hours, then alert. This
   filters out transient noise.

3. **Adaptive thresholds:** Instead of fixed thresholds, use rolling baselines. Alert
   when a metric is N standard deviations from its 30-day average. This adapts to
   seasonal patterns.

4. **Alert cooldown:** After alerting, suppress the same alert for a configurable period
   (e.g., 4 hours) to prevent alert storms.

### The Retraining Decision

When drift or degradation is detected, you have three options:

**Option 1: Investigate**
- When: moderate drift (PSI 0.10-0.20), performance slightly below threshold.
- Action: examine the drift report, identify which features drifted, trace back to
  root cause (data pipeline change? seasonal pattern? real distribution shift?).
- Timeline: resolve within 1-3 days.

**Option 2: Retrain**
- When: significant drift (PSI > 0.20 in multiple features), performance clearly
  degraded but model structure is still appropriate.
- Action: retrain the model on recent data. Use the existing pipeline. Validate on
  holdout set. Deploy through the normal CI/CD process.
- Timeline: hours to days depending on pipeline maturity.

**Option 3: Rollback**
- When: severe performance degradation, especially if a recent deployment caused it.
- Action: rollback to the previous model version. This is why model versioning
  (MLflow model registry, SageMaker model registry) is critical.
- Timeline: minutes.

**Decision flowchart:**
```
Drift detected?
  |
  +-- No --> Continue monitoring
  |
  +-- Yes --> Performance degraded?
                |
                +-- No --> Log, investigate, monitor closely
                |
                +-- Yes --> Recent deployment?
                              |
                              +-- Yes --> Rollback, then investigate
                              |
                              +-- No --> Retrain with recent data
```

### A/B Testing Models in Production

When you have a new model version (v2) that you want to deploy, do not just replace
the old model (v1). Use A/B testing:

1. Route 95% of traffic to v1 (the champion), 5% to v2 (the challenger).
2. Compare metrics on both populations.
3. If v2 performs better (with statistical significance), gradually increase its traffic
   share: 5% -> 10% -> 25% -> 50% -> 100%.
4. If v2 performs worse, kill it and investigate.

**Implementation:** Use a traffic router (e.g., AWS SageMaker's production variants,
or a custom router with feature flags) that splits requests by a hash of the user ID.
Hashing ensures the same user always sees the same model version (consistency).

### Shadow Mode Deployment

Before even A/B testing, you can deploy v2 in "shadow mode":

1. Both v1 and v2 receive every request.
2. Only v1's predictions are returned to the user.
3. v2's predictions are logged but not served.
4. Compare v1 and v2 predictions offline.

**Advantages:**
- Zero risk to users. v2's bad predictions never reach anyone.
- Full traffic comparison (not just 5%).
- Can detect systematic differences (e.g., v2 always predicts higher confidence).

**Disadvantages:**
- Doubles compute cost (running two models).
- Does not test v2's real-world impact (e.g., how users react to different predictions).

### Netflix's Approach to Model Monitoring (Real Example)

Netflix uses ML extensively for recommendations, content decisions, and operations.
Their monitoring approach:

1. **Metaflow + dashboards:** Every model has an automatically generated dashboard
   showing prediction distributions, feature importance, and performance metrics over
   time.

2. **Automated canary analysis:** New model versions are deployed as canaries. Netflix's
   Kayenta system automatically compares the canary's metrics against the baseline and
   decides whether to promote or rollback.

3. **A/B testing everything:** Every model change goes through an A/B test. They run
   thousands of simultaneous A/B tests.

4. **Feature monitoring:** They monitor not just model outputs but individual feature
   pipelines. If a feature pipeline has latency or quality issues, they can fall back
   to cached values or simpler features.

5. **Business metric correlation:** Model metrics (AUC, precision) are correlated with
   business metrics (user engagement, watch time). A model might have great AUC but
   not improve watch time, which means the model is solving the wrong problem.

### Summary: The Complete Monitoring Stack

```
+------------------+    +------------------+    +------------------+
| Data Monitoring  |    | Model Monitoring |    | System Monitoring|
| - Schema checks  |    | - Drift detection|    | - Latency (p99)  |
| - Null rates     |    | - PSI / KS tests |    | - Error rates    |
| - Volume checks  |    | - Performance    |    | - CPU / Memory   |
| - Freshness      |    |   metrics        |    | - Request volume |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         v                       v                       v
+------------------------------------------------------------------+
|                     Alert Aggregation Layer                       |
|  (CloudWatch / Datadog / Grafana + PagerDuty / Slack)            |
+------------------------------------------------------------------+
         |                       |                       |
         v                       v                       v
+------------------+    +------------------+    +------------------+
| Auto-remediation |    | Human Review     |    | Escalation       |
| - Restart service|    | - Check dashboard|    | - Page on-call   |
| - Cache fallback |    | - Analyze report |    | - Rollback model |
+------------------+    +------------------+    +------------------+
```

### Interview Tips for Monitoring Questions

**Q: "How would you detect data drift in production?"**
A: "I would implement statistical tests comparing current data distributions against a
reference (training data) snapshot. Specifically, the Kolmogorov-Smirnov test for
statistical significance and Population Stability Index for magnitude. I would run these
hourly on a rolling window of production data and publish results to CloudWatch with
tiered alerting: info for PSI 0.10-0.20, critical for PSI > 0.20 across multiple
features."

**Q: "What is the difference between data drift and concept drift?"**
A: "Data drift means P(X) changed -- the input distributions shifted. Concept drift
means P(Y|X) changed -- the relationship between inputs and outputs shifted. You can
detect data drift without ground truth labels. Concept drift requires ground truth,
which in many systems (like fraud detection) is delayed. That is why monitoring both
data drift (as an early warning) and performance metrics (as ground truth becomes
available) is essential."

**Q: "Your model's precision dropped from 0.95 to 0.60 overnight. Walk me through
your investigation."**
A: "First, I would check if there was a recent deployment (rollback immediately if so).
Second, I would examine the drift report to see if input distributions changed. Third,
I would check upstream data pipelines for anomalies (null rates, schema changes, volume
drops). Fourth, I would look at which predictions changed -- is it a specific segment
of users or across the board? Fifth, once I identify the root cause, I would decide
between fixing the data pipeline, retraining the model, or rolling back to a previous
version."

---

**Key files referenced in this guide:**
- `src/monitoring/drift_detection.py` -- DriftDetector class, KS test, PSI calculation
- `src/monitoring/performance.py` -- PerformanceMonitor class, metric tracking, alerting
