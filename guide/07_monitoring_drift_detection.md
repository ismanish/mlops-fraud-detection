# 07 — ML Monitoring & Drift Detection

## Table of Contents
1. [Why ML Models Degrade in Production](#why-ml-models-degrade)
2. [Types of Drift](#types-of-drift)
3. [Statistical Tests for Drift Detection](#statistical-tests)
4. [Our Drift Detection Implementation](#our-implementation)
5. [Performance Monitoring](#performance-monitoring)
6. [CloudWatch Custom Metrics and Dashboards](#cloudwatch-metrics-and-dashboards)
7. [Alerting Strategy](#alerting-strategy)
8. [Evidently AI for Monitoring](#evidently-ai)
9. [Interview Questions](#interview-questions)

---

## Why ML Models Degrade in Production

Traditional software is deterministic: if the code does not change, the output does not
change. ML models are fundamentally different -- their quality depends on the
relationship between the data they were trained on and the data they receive in
production. When this relationship drifts, model performance degrades silently.

### The ML Model Decay Curve

```
Model
Quality
  ^
  |  ****
  |      ****
  |          ****
  |              ****                 <-- Gradual degradation
  |                  *****
  |                       *****
  |                            ****
  |                                ***   <-- Sudden shift (e.g., COVID)
  |                                   *
  +---------------------------------------------> Time
  |   Training    |  Production  |  Degraded  |
  |   Period      |  (stable)    |  (retrain) |
```

### Root Causes of Model Degradation

**1. Data Drift (Covariate Shift)**
The distribution of input features changes. Example: transaction amounts increase
due to inflation, or a new payment method introduces novel feature patterns.

**2. Concept Drift**
The relationship between features and target changes. Example: fraudsters change
tactics -- what used to be a fraudulent pattern is now legitimate, or new fraud
patterns emerge that the model has never seen.

**3. Upstream Data Issues**
A data pipeline breaks, a feature starts returning nulls, a third-party API changes
its format, or a feature engineering bug is introduced.

**4. Population Shift**
The user population changes. Example: expanding from US to European customers
introduces different spending patterns.

**5. Feedback Loops**
The model's own predictions influence future data. Example: if the fraud model blocks
a legitimate transaction, the customer may abandon the transaction, creating a
pattern the model reinforces.

### Why This Matters for Fraud Detection

Fraud detection is especially vulnerable to concept drift because it is an
adversarial domain. Fraudsters actively adapt to evade detection. A model trained
on historical fraud patterns becomes less effective as criminals develop new
techniques. This is why continuous monitoring and retraining are not optional --
they are fundamental requirements.

---

## Types of Drift

### Data Drift (Feature Drift / Covariate Shift)

The marginal distribution of input features P(X) changes, while the conditional
distribution P(Y|X) may or may not change.

```
Training Distribution        Production Distribution
      ___                           ___
     /   \                         /   \
    /     \                       /     \
   /       \                   /         \
  /         \                 /           \
_/           \_             _/             \_
   Amount ($)                  Amount ($)
   Mean: $88                   Mean: $140
```

**Detection method:** Compare feature distributions using KS test, PSI, or
Jensen-Shannon divergence.

**Our implementation detects this** by comparing training data distributions
against production data distributions for each feature.

### Concept Drift

The conditional distribution P(Y|X) changes. The same input features now map to
different outcomes. The model's learned decision boundary is wrong.

```
   Feature Space

   Before Concept Drift          After Concept Drift

   . . . x x x                  . . . x . .
   . . x x x x                  . x x x . .
   . . . . x x     ------>      . . x x x .
   . . . . . x                  . . . x x x
   . . . . . .                  . . . . x x

   (x = fraud, . = legit)       (fraud pattern shifted)
   Decision boundary is         Old boundary misclassifies
   correct                      many transactions
```

**Detection method:** Monitor model performance metrics (precision, recall, F1)
over time. If ground truth labels are available, compare predicted vs actual.

**Types of concept drift:**
- **Sudden** -- Abrupt change (e.g., new regulation, pandemic)
- **Gradual** -- Slow transition between concepts
- **Incremental** -- Steady, continuous change
- **Recurring** -- Seasonal patterns (e.g., holiday spending)

### Prediction Drift (Output Drift)

The distribution of model predictions P(Y_hat) changes. Even if individual
predictions look reasonable, a shift in the overall distribution signals trouble.

```
Training Period:     5% of predictions are "fraud"
Week 1 Production:   5.2% fraud  (normal variance)
Week 2 Production:   4.8% fraud  (normal variance)
Week 3 Production:   8.1% fraud  (ALERT: prediction drift)
Week 4 Production:  12.3% fraud  (ALERT: significant shift)
```

**Detection method:** Track `prediction_positive_rate` over time. Our FastAPI app
publishes `FraudPredicted` to CloudWatch for exactly this purpose.

### Label Drift (Prior Probability Shift)

The distribution of the target variable P(Y) changes. The actual fraud rate in the
real world shifts.

```
Training data:     0.17% fraud rate
Production Q1:     0.19% fraud rate   (normal)
Production Q2:     0.45% fraud rate   (ALERT: label drift)
```

**Detection method:** Requires ground truth labels (confirmed fraud from
investigations). Compare the actual fraud rate against the training baseline.

### Summary Table

```
+-------------------+------------------+-----------------------+------------------+
| Drift Type        | What Changes     | How to Detect         | Ground Truth     |
|                   |                  |                       | Required?        |
+-------------------+------------------+-----------------------+------------------+
| Data Drift        | P(X)             | KS test, PSI, JS div | No               |
| Concept Drift     | P(Y|X)           | Performance metrics   | Yes              |
| Prediction Drift  | P(Y_hat)         | Prediction rate shift | No               |
| Label Drift       | P(Y)             | Actual outcome rate   | Yes              |
+-------------------+------------------+-----------------------+------------------+
```

---

## Statistical Tests for Drift Detection

### Kolmogorov-Smirnov (KS) Test

The KS test measures the maximum distance between two empirical cumulative
distribution functions (CDFs). It is non-parametric -- it makes no assumptions about
the underlying distribution.

```
CDF
  ^
1 |                    .........****
  |               ....*****
  |            ...**
  |          ..**
  |        ..*         D = max distance between CDFs
  |       .*    <------|
  |      .*
  |    .*
  |  .*
0 +--**----------------------------> Value
       Reference CDF (...)
       Current CDF (****)
```

**Interpretation:**
- KS statistic (D): 0 to 1, higher = more different
- p-value < threshold (0.05): reject null hypothesis, distributions are different
- p-value >= threshold: no evidence of drift

**Our implementation:**

```python
from scipy import stats

ks_stat, ks_pvalue = stats.ks_2samp(
    self.reference[col],   # Training data for feature
    current_data[col]      # Production data for feature
)
is_drifted = ks_pvalue < self.threshold  # threshold = 0.05
```

**Strengths:** Works for any continuous distribution, sensitive to changes in shape,
location, and spread.

**Weaknesses:** Sensitive to sample size (large samples always detect tiny
differences), only works for univariate continuous data.

### Population Stability Index (PSI)

PSI quantifies how much a distribution has shifted from a reference. Originally
developed in credit risk modeling -- very relevant to our domain.

```
PSI = SUM[ (current_pct_i - reference_pct_i) * ln(current_pct_i / reference_pct_i) ]
```

**Interpretation:**
```
+-------------+----------------------------+
| PSI Value   | Interpretation             |
+-------------+----------------------------+
| PSI < 0.1   | No significant shift       |
| 0.1 - 0.25  | Moderate shift, investigate|
| PSI > 0.25  | Significant shift, action  |
+-------------+----------------------------+
```

**Our implementation:**

```python
def _calculate_psi(self, reference: pd.Series, current: pd.Series,
                   bins: int = 10) -> float:
    """Population Stability Index -- measures distribution shift."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # Clip to avoid log(0)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
```

**Key implementation detail:** We use percentile-based binning from the reference
distribution. This ensures bins are evenly populated in the reference, making PSI
sensitive to changes across the entire range, not just the tails.

### Chi-Squared Test

Used for categorical features. Compares observed frequencies against expected
frequencies derived from the reference distribution.

```
chi2 = SUM[ (observed_i - expected_i)^2 / expected_i ]
```

Not implemented in our project because all features are continuous (PCA components
and Amount), but important to know for interviews.

### Jensen-Shannon Divergence (JSD)

A symmetric, bounded version of KL divergence. Measures similarity between two
probability distributions.

```
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
where M = 0.5 * (P + Q)
```

**Properties:**
- Bounded: 0 (identical) to 1 (completely different, when using log base 2)
- Symmetric: JSD(P||Q) = JSD(Q||P), unlike KL divergence
- Always defined: Unlike KL divergence, it does not require Q(x) > 0 wherever P(x) > 0

**When to use JSD over KS test:** JSD is more robust to sample size differences and
provides a more interpretable bounded score. The KS test gives a p-value that depends
on sample size, making it hard to compare across features with different cardinalities.

---

## Our Drift Detection Implementation

### File: `src/monitoring/drift_detection.py`

The `DriftDetector` class implements the complete drift detection pipeline.

### Architecture

```
+-------------------+       +-------------------+
| Reference Data    |       | Current Data      |
| (X_train.csv)     |       | (X_test.csv or    |
|                   |       |  production batch) |
+--------+----------+       +--------+----------+
         |                           |
         v                           v
+--------+---------------------------+----------+
|              DriftDetector                     |
|                                                |
|  For each numeric feature:                     |
|    1. KS test (scipy.stats.ks_2samp)          |
|    2. PSI calculation (percentile binning)     |
|    3. Compare means (ref vs current)           |
|    4. Flag if p-value < threshold              |
|                                                |
+--------+--------------------------------------+
         |
         v
+--------+----------+
| Drift Report      |
| {                 |
|   "drifted": T/F, |
|   "features": {   |
|     "V1": {...},   |
|     "V14": {...},  |
|   },              |
|   "drift_ratio":  |
|     0.1034        |
| }                 |
+--------+----------+
         |
    +----+----+
    |         |
    v         v
+-------+ +----------+
| JSON  | | CloudWatch|
| File  | | Alarm     |
+-------+ +----------+
```

### DriftDetector Class Walkthrough

**Initialization:**

```python
class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)
```

The reference data is the training set. The threshold (0.05) is the p-value cutoff
for the KS test -- matching the conventional alpha level for statistical significance.
This is configurable via `params.yaml`:

```yaml
monitoring:
  drift_threshold: 0.05
```

**Core Detection Logic:**

```python
def detect_drift(self, current_data: pd.DataFrame) -> dict:
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": {},
        "drifted": False
    }

    for col in self.reference.select_dtypes(include=[np.number]).columns:
        if col not in current_data.columns:
            continue

        ks_stat, ks_pvalue = stats.ks_2samp(self.reference[col], current_data[col])
        psi = self._calculate_psi(self.reference[col], current_data[col])

        is_drifted = ks_pvalue < self.threshold
        results["features"][col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": round(float(psi), 6),
            "drifted": is_drifted,
            "ref_mean": round(float(self.reference[col].mean()), 6),
            "cur_mean": round(float(current_data[col].mean()), 6),
        }
        if is_drifted:
            results["drifted"] = True
```

For each numeric feature, we run two tests:
1. **KS test** -- The primary drift signal. If p-value < 0.05, the feature is flagged.
2. **PSI** -- Supplementary metric for interpretability. PSI gives a magnitude score,
   while KS test gives a binary yes/no.

The results include both `ref_mean` and `cur_mean` so we can see the direction and
magnitude of the shift at a glance.

**Aggregate Drift Metrics:**

```python
n_drifted = sum(1 for f in results["features"].values() if f["drifted"])
results["n_features_drifted"] = n_drifted
results["n_features_total"] = len(results["features"])
results["drift_ratio"] = round(n_drifted / max(len(results["features"]), 1), 4)
```

The `drift_ratio` is the fraction of features showing drift. This is important
because with 29 features and alpha=0.05, we expect ~1.45 features to show
"drift" by random chance (false positives). A drift_ratio of 0.05 is normal.
A drift_ratio of 0.3+ signals real distributional shift.

### Running Drift Detection

```python
def run_drift_detection():
    params = load_params()
    root = get_project_root()
    processed_dir = root / params["data"]["processed_path"]

    X_train = pd.read_csv(processed_dir / "X_train.csv")   # Reference
    X_test = pd.read_csv(processed_dir / "X_test.csv")     # "Production" proxy

    detector = DriftDetector(
        reference_data=X_train,
        threshold=params["monitoring"]["drift_threshold"],
    )
    results = detector.detect_drift(X_test)
```

In our project, `X_test` acts as a proxy for production data. In a real deployment,
you would collect a batch of recent production predictions and compare them against
the training distribution.

### CloudWatch Alarm on Drift

```python
def _send_drift_alarm(results: dict, params: dict):
    cloudwatch = boto3.client("cloudwatch", region_name=aws_config["region"])
    cloudwatch.put_metric_data(
        Namespace="MLOps/FraudDetection",
        MetricData=[
            {"MetricName": "DataDriftDetected", "Value": 1, "Unit": "Count"},
            {"MetricName": "DriftedFeatureRatio", "Value": results["drift_ratio"],
             "Unit": "None"},
        ],
    )
```

Two metrics:
- `DataDriftDetected` -- Binary signal (1 = drift found). Triggers a CloudWatch alarm.
- `DriftedFeatureRatio` -- Continuous signal (0.0 to 1.0). Tracks severity over time.

---

## Performance Monitoring

### File: `src/monitoring/performance.py`

Performance monitoring tracks whether the model's predictions are still accurate.
Unlike drift detection (which does not need labels), performance monitoring requires
ground truth labels.

### PerformanceMonitor Class

```python
class PerformanceMonitor:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def evaluate(self, y_true, y_pred, y_prob) -> dict:
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

**Metrics tracked:**

| Metric | Why It Matters for Fraud Detection |
|--------|-----------------------------------|
| Precision | Of transactions flagged as fraud, how many actually are? Low precision = too many false alarms, blocking legitimate customers |
| Recall | Of actual fraud cases, how many did we catch? Low recall = fraud slipping through |
| F1 | Harmonic mean of precision and recall. Single number for overall quality |
| AUC-ROC | Threshold-independent measure of model discrimination ability |
| prediction_positive_rate | Fraction of predictions that are "fraud". Tracks prediction drift |

### Threshold-Based Alerting

```python
alerts = []
if metrics["recall"] < self.thresholds["min_recall"]:
    alerts.append(f"recall {metrics['recall']:.4f} < {self.thresholds['min_recall']}")
if metrics["precision"] < self.thresholds["min_precision"]:
    alerts.append(f"precision {metrics['precision']:.4f} < {self.thresholds['min_precision']}")
if metrics["f1"] < self.thresholds["min_f1"]:
    alerts.append(f"f1 {metrics['f1']:.4f} < {self.thresholds['min_f1']}")
```

Thresholds from `params.yaml`:

```yaml
thresholds:
  min_recall: 0.80
  min_precision: 0.50
  min_f1: 0.60
  min_auc_roc: 0.95
```

**Why recall threshold (0.80) is higher than precision (0.50):**
In fraud detection, missing actual fraud (false negatives, low recall) is more
costly than flagging a legitimate transaction (false positives, low precision).
A missed fraud case means direct financial loss. A false positive only causes
customer inconvenience (a phone call to verify).

### Performance Report Flow

```
+------------------+       +--------------------+       +-----------------+
| eval_metrics.json| ----> | PerformanceMonitor | ----> | perf_YYYYMMDD   |
| (from evaluate)  |       | .evaluate()        |       | _HHMMSS.json    |
+------------------+       +--------+-----------+       +-----------------+
                                     |
                               +-----+-----+
                               |           |
                         threshold    threshold
                           PASS         FAIL
                               |           |
                               v           v
                          +--------+  +-----------+
                          | Log    |  | CloudWatch|
                          | "OK"   |  | Alarm:    |
                          +--------+  | ModelDeg- |
                                      | raded = 1 |
                                      +-----------+
```

---

## CloudWatch Custom Metrics and Dashboards

### Custom Metric Namespace

All our metrics are published under the namespace `MLOps/FraudDetection`. This
separates them from AWS-managed metrics (like `AWS/Lambda`).

### Metrics We Publish

```
Namespace: MLOps/FraudDetection
+------------------------+------------+---------------------------+
| Metric Name            | Source     | Purpose                   |
+------------------------+------------+---------------------------+
| PredictionLatency      | app.py     | Track inference speed     |
| FraudPredicted         | app.py     | Count fraud predictions   |
| FraudProbability       | app.py     | Track prediction dist.    |
| DataDriftDetected      | drift_det. | Binary drift signal       |
| DriftedFeatureRatio    | drift_det. | Drift severity (0-1)      |
| ModelDegraded          | perf.py    | Binary degradation signal |
+------------------------+------------+---------------------------+
```

### Our CloudWatch Dashboard

Defined in `infrastructure/terraform/cloudwatch.tf`:

```
+------------------------------------------------------------------+
|                  MLOps Fraud Detection Dashboard                  |
+--------------------------------+---------------------------------+
|  Prediction Latency (5 min)    |  Fraud Detection Rate (5 min)   |
|                                |                                 |
|  ___       ___                 |      _                          |
| /   \     /   \               |     / \                         |
|/     \___/     \___           |    /   \    ___                  |
|                               |___/     \__/   \___             |
|  avg: 45ms, p99: 120ms       |  sum: 23 fraud predictions      |
+--------------------------------+---------------------------------+
|  Data Drift Detected (1 hr)   |  Model Degradation (1 hr)       |
|                                |                                 |
|  ___                           |                                 |
| |   |                          |  No alarms                     |
| |   |                          |                                 |
|_|   |________________________ |_________________________________|
|  1 drift event                |  0 degradation events           |
+--------------------------------+---------------------------------+
|  Lambda Invocations (5 min)   |  Lambda Duration (5 min)        |
|                                |                                 |
|     ___                        |    ___                          |
|    /   \       ___             |   /   \                         |
|___/     \_____/   \___        |__/     \______                  |
|                               |                                  |
|  Invocations (blue)           |  avg: 200ms                     |
|  Errors (red)                 |  (includes cold starts)         |
+--------------------------------+---------------------------------+
```

Six widgets tracking three categories:
1. **ML metrics** -- Prediction latency, fraud rate, drift, degradation
2. **Infrastructure metrics** -- Lambda invocations and errors
3. **Performance metrics** -- Lambda duration (includes cold starts)

### CloudWatch Alarms

Two alarms configured in Terraform:

**1. High Error Rate Alarm**

```hcl
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2           # Must breach 2 consecutive periods
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300         # 5-minute windows
  statistic           = "Sum"
  threshold           = 10          # >10 errors in 5 minutes
  treat_missing_data  = "notBreaching"
}
```

Triggers when Lambda has more than 10 errors in two consecutive 5-minute periods.
`treat_missing_data = "notBreaching"` prevents false alarms when the function is
not being invoked (no data = no errors).

**2. Drift Detection Alarm**

```hcl
resource "aws_cloudwatch_metric_alarm" "drift_detected" {
  alarm_name          = "${var.project_name}-drift-detected"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1           # Single occurrence triggers
  metric_name         = "DataDriftDetected"
  namespace           = "MLOps/FraudDetection"
  period              = 3600        # 1-hour window
  statistic           = "Sum"
  threshold           = 0           # Any drift triggers alarm
}
```

Triggers immediately on any drift detection. This alarm would typically be connected
to an SNS topic that emails the ML team or posts to a Slack channel.

---

## Alerting Strategy

### Decision Matrix: When to Retrain vs Investigate

```
                        Performance Degraded?
                        No              Yes
                  +----------------+------------------+
   Drift          |                |                  |
   Detected?  No  |   All Good     |   Investigate    |
                  |   (continue    |   (upstream bug? |
                  |    monitoring) |    label issue?) |
                  +----------------+------------------+
              Yes |   Monitor      |   Retrain        |
                  |   Closely      |   (confirmed     |
                  |   (drift may   |    degradation   |
                  |    be benign)  |    from drift)   |
                  +----------------+------------------+
```

### Detailed Response Playbook

**Scenario 1: Data drift detected, performance OK**
- Action: Increase monitoring frequency, log the drift report
- Rationale: Drift does not always cause degradation. New legitimate transaction
  patterns may shift feature distributions without affecting fraud detection accuracy.
- Timeline: Review in 1 week

**Scenario 2: No drift detected, performance degraded**
- Action: Investigate immediately
- Possible causes: Upstream data pipeline issue, labeling error, feature engineering
  bug, or concept drift that does not manifest as data drift
- Timeline: Investigate within 24 hours

**Scenario 3: Drift detected AND performance degraded**
- Action: Trigger retraining pipeline
- This is the clearest signal: the world has changed, and the model has not kept up
- Timeline: Retrain immediately, deploy within hours

**Scenario 4: Everything normal**
- Action: Continue monitoring
- Run drift detection daily, performance monitoring weekly (or when labels arrive)

### Alert Severity Levels

```
+----------+----------------------------+-------------------+-----------+
| Severity | Condition                  | Notification      | Response  |
+----------+----------------------------+-------------------+-----------+
| P0       | Lambda error rate > 50%    | PagerDuty         | Immediate |
| P1       | Recall < 0.60              | Slack + Email     | 1 hour    |
| P1       | drift_ratio > 0.50         | Slack + Email     | 1 hour    |
| P2       | Recall < 0.80              | Email             | 24 hours  |
| P2       | drift_ratio > 0.20         | Email             | 24 hours  |
| P3       | Latency p99 > 500ms        | Dashboard only    | Next sprint|
| P3       | Any single feature drifted | Dashboard only    | Next sprint|
+----------+----------------------------+-------------------+-----------+
```

### Retraining Trigger Pipeline

```
CloudWatch Alarm         SNS Topic          Lambda / Step Functions
+-----------+       +------------+       +---------------------+
| Drift +   | ----> | Notify ML  | ----> | Trigger GitHub      |
| Degraded  |       | Team       |       | Actions workflow    |
+-----------+       +------------+       | (workflow_dispatch) |
                                          +--------+------------+
                                                   |
                                          +--------v------------+
                                          | train.yml           |
                                          | 1. Ingest new data  |
                                          | 2. Preprocess       |
                                          | 3. Train model      |
                                          | 4. Evaluate         |
                                          | 5. Quality gates    |
                                          +--------+------------+
                                                   |
                                          +--------v------------+
                                          | deploy.yml          |
                                          | 1. Build container  |
                                          | 2. Push to ECR      |
                                          | 3. Update Lambda    |
                                          | 4. Smoke test       |
                                          +---------------------+
```

---

## Evidently AI for Monitoring

Evidently is an open-source ML monitoring library that provides pre-built reports
for data drift, model performance, and data quality. It is included in our
`requirements.txt`.

### Why Evidently Complements Our Custom Implementation

Our custom drift detection (`DriftDetector`) is lightweight and production-focused.
Evidently adds:

1. **Rich HTML reports** -- Visual dashboards for exploring drift feature-by-feature
2. **Multiple statistical tests** -- Wasserstein distance, Anderson-Darling, Cramér's V
3. **Target drift** -- Tracks changes in the target distribution
4. **Data quality** -- Detects missing values, duplicates, new categories
5. **Classification reports** -- Confusion matrix evolution over time

### Example Usage with Our Data

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import pandas as pd

# Load reference and current data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Create a drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)

# Save as HTML for visual inspection
report.save_html("metrics/drift/evidently_report.html")

# Or extract results as a dictionary for programmatic use
results = report.as_dict()
drift_detected = results["metrics"][0]["result"]["dataset_drift"]
```

### Evidently vs Our Custom Implementation

```
+-----------------------+---------------------+---------------------+
| Feature               | Our DriftDetector   | Evidently           |
+-----------------------+---------------------+---------------------+
| KS test               | Yes                 | Yes                 |
| PSI                   | Yes                 | Yes                 |
| Jensen-Shannon        | No                  | Yes                 |
| Wasserstein           | No                  | Yes                 |
| HTML reports          | No                  | Yes                 |
| CloudWatch publish    | Yes                 | No (custom needed)  |
| Lightweight (Lambda)  | Yes (~0 overhead)   | No (heavy imports)  |
| Categorical features  | No (not needed)     | Yes                 |
| Real-time (per req.)  | No (batch only)     | No (batch only)     |
+-----------------------+---------------------+---------------------+
```

For production Lambda deployment, our lightweight implementation is preferred.
For offline analysis and reporting, Evidently provides richer insights.

---

## Interview Questions

### Q1: What is the difference between data drift and concept drift?

**A:** Data drift (covariate shift) is when the distribution of input features P(X)
changes -- for example, average transaction amounts increasing due to inflation or a
new customer segment emerging. Concept drift is when the relationship P(Y|X) between
features and outcomes changes -- for example, fraudsters adopting new tactics so that
previously legitimate-looking patterns become fraudulent. Data drift can be detected
without labels by comparing feature distributions, while concept drift requires
ground truth labels to detect because the inputs may look the same but map to
different outcomes. In our project, we detect data drift using KS tests and PSI on
each feature, comparing training data distributions against production data.

### Q2: Why do you use the KS test instead of just comparing means?

**A:** Comparing means only detects shifts in central tendency. Two distributions can
have identical means but completely different shapes, spreads, or modalities. The KS
test compares the entire cumulative distribution function, capturing changes in any
aspect of the distribution -- shifts, spreads, skewness, or the emergence of new
modes. For example, if a bimodal distribution replaces a unimodal one with the same
mean, comparing means would miss it entirely, but the KS test would detect it. We
supplement the KS test with PSI because the KS test's p-value is sensitive to sample
size (large samples detect trivially small differences), while PSI provides an
interpretable magnitude score independent of sample size.

### Q3: What is PSI and what are the standard interpretation thresholds?

**A:** Population Stability Index (PSI) measures how much a variable's distribution
has shifted from a reference. It is calculated by binning both distributions,
computing the proportion difference in each bin, and summing the product of that
difference with the log ratio. PSI < 0.1 indicates no significant shift, 0.1-0.25
indicates moderate shift warranting investigation, and PSI > 0.25 indicates
significant shift requiring action. PSI originated in credit risk modeling and is
especially relevant to our fraud detection domain. In our implementation, we use
percentile-based binning from the reference distribution to ensure even bin
populations, and we clip proportions to 1e-6 to avoid log(0) issues.

### Q4: Why do you monitor prediction distribution, not just accuracy?

**A:** Accuracy (and related metrics like F1, precision, recall) requires ground truth
labels, which in fraud detection are delayed -- you only know a transaction was truly
fraudulent after investigation, which can take days or weeks. Prediction distribution
monitoring (tracking the rate and distribution of "fraud" predictions) is available
immediately and can signal problems before labels arrive. If our model suddenly
predicts 15% of transactions as fraud when the training baseline was 0.17%, something
is wrong -- either the model is miscalibrated, the input distribution shifted, or
there is an upstream data issue. Our FastAPI app publishes `FraudPredicted` to
CloudWatch for exactly this kind of real-time prediction drift monitoring.

### Q5: How would you handle drift in a feature that is actually improving model performance?

**A:** Not all drift is harmful. If a feature distribution shifts but model
performance remains stable or improves, the drift is benign -- it may reflect natural
evolution in the customer base. Our alerting strategy uses a 2x2 decision matrix:
drift without degradation triggers close monitoring but not retraining. Only when
drift coincides with measurable performance degradation do we trigger retraining.
This prevents unnecessary retraining cycles that waste compute resources and introduce
risk (every retraining is a chance to introduce a worse model). The key insight is
that drift detection tells you the world changed; performance monitoring tells you
whether that change matters.

### Q6: Your drift detection runs on X_test. How would you do this in production?

**A:** In production, I would collect a rolling window of recent prediction requests
(e.g., the last 24 hours of transactions) and compare that batch against the training
reference distribution stored in S3. The collection could be done via a Lambda that
logs each request to S3 or Kinesis, and a scheduled job (CloudWatch Events + Lambda
or Step Functions) that runs drift detection daily. The reference data would be
updated whenever the model is retrained. For real-time drift detection, you could use
a streaming approach with Kinesis Data Analytics, maintaining running statistics
(means, histograms) and comparing them against reference statistics on each new data
point, but this adds significant complexity.

### Q7: Explain the multiple-testing problem in drift detection.

**A:** With 29 features and a significance level of alpha=0.05, we expect
29 * 0.05 = 1.45 features to show "significant" drift by pure random chance (false
positives). This is the multiple-testing problem. Solutions include Bonferroni
correction (use alpha/n = 0.05/29 = 0.0017 per feature, but very conservative),
Benjamini-Hochberg procedure (controls the false discovery rate), or our approach:
computing `drift_ratio` and only alarming when a substantial fraction of features
drift simultaneously. A drift_ratio of 0.05 (1-2 features) is expected noise; a
drift_ratio of 0.30+ (9+ features) is almost certainly real drift. This aggregate
approach is more robust than per-feature alarming.

### Q8: What metrics are most important for monitoring a fraud detection model?

**A:** The most critical metric is recall (sensitivity) because a missed fraud case
means direct financial loss. Our threshold is min_recall=0.80, meaning we require
catching at least 80% of fraud. Precision matters too but is secondary -- false
positives cause customer friction (blocked transactions) but not financial loss. Our
threshold is min_precision=0.50. AUC-ROC is the most stable metric across threshold
choices and is our quality gate for deployment (min_auc_roc=0.95). Beyond traditional
metrics, we track prediction_positive_rate (fraction of "fraud" predictions) as a
proxy for prediction drift, and latency to ensure the model serves within SLA.
Average precision (PR-AUC) is especially important for imbalanced datasets like ours
because it focuses on the positive class performance.

### Q9: How would you implement a model performance dashboard for stakeholders?

**A:** Our CloudWatch dashboard serves engineering needs, but stakeholders need
business metrics. I would build a dashboard showing: (1) daily fraud detection rate
and dollar amount saved, (2) false positive rate and customer impact (blocked
legitimate transactions), (3) model confidence distribution (how certain is the model
about its predictions), (4) trend lines showing these metrics over weeks/months with
automated anomaly highlighting. For implementation, I would use CloudWatch Insights
queries feeding into a Grafana dashboard, or export metrics to a data warehouse and
use Looker/Tableau. The key is translating ML metrics into business language --
"recall dropped from 85% to 78%" becomes "we are missing 7% more fraud cases, costing
approximately $X per month."

### Q10: What is the difference between online and offline monitoring?

**A:** Online monitoring happens in real-time as predictions are served. Our FastAPI
app publishing metrics to CloudWatch on every request is online monitoring -- we can
see latency spikes, prediction distribution changes, and error rates immediately.
Offline monitoring happens on batched data, typically on a schedule. Our
`run_drift_detection()` and `run_performance_monitoring()` functions are offline
monitoring -- they process a batch of data and compare it against references.
Online monitoring catches sudden failures (model crashes, extreme latency), while
offline monitoring catches gradual degradation (slow drift, subtle performance loss).
A production system needs both: online for operational health, offline for statistical
rigor.
