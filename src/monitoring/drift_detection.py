"""Data and model drift detection using statistical tests.

Compares current (production) data distribution against the training
reference distribution to detect feature drift and prediction drift.
"""

import json
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import get_aws_config, get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        return {
            col: {"mean": df[col].mean(), "std": df[col].std(), "median": df[col].median()}
            for col in df.select_dtypes(include=[np.number]).columns
        }

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        results = {"timestamp": datetime.utcnow().isoformat(), "features": {}, "drifted": False}

        for col in self.reference.select_dtypes(include=[np.number]).columns:
            if col not in current_data.columns:
                continue

            ks_stat, ks_pvalue = stats.ks_2samp(self.reference[col], current_data[col])
            psi = self._calculate_psi(self.reference[col], current_data[col])

            is_drifted = bool(ks_pvalue < self.threshold)
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

        n_drifted = sum(1 for f in results["features"].values() if f["drifted"])
        results["n_features_drifted"] = n_drifted
        results["n_features_total"] = len(results["features"])
        results["drift_ratio"] = round(n_drifted / max(len(results["features"]), 1), 4)

        return results

    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Population Stability Index — measures distribution shift."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)

        return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def run_drift_detection():
    params = load_params()
    root = get_project_root()
    processed_dir = root / params["data"]["processed_path"]

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")

    detector = DriftDetector(
        reference_data=X_train,
        threshold=params["monitoring"]["drift_threshold"],
    )
    results = detector.detect_drift(X_test)

    reports_dir = root / "metrics" / "drift"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    if results["drifted"]:
        logger.warning(
            f"DRIFT DETECTED: {results['n_features_drifted']}"
            f"/{results['n_features_total']} features"
        )
        _send_drift_alarm(results, params)
    else:
        logger.info("No drift detected")

    logger.info(f"Drift report saved to {report_path}")
    return results


def _send_drift_alarm(results: dict, params: dict):
    try:
        aws_config = get_aws_config(params)
        cloudwatch = boto3.client("cloudwatch", region_name=aws_config["region"])
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
        logger.info("Drift alarm published to CloudWatch")
    except Exception as e:
        logger.warning(f"Failed to publish drift alarm: {e}")


if __name__ == "__main__":
    run_drift_detection()
