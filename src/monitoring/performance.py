"""Model performance monitoring: tracks production metrics over time."""

import json
from datetime import datetime

import boto3
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.utils.config import get_aws_config, get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
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

        alerts = []
        if metrics["recall"] < self.thresholds["min_recall"]:
            alerts.append(f"recall {metrics['recall']:.4f} < {self.thresholds['min_recall']}")
        if metrics["precision"] < self.thresholds["min_precision"]:
            min_prec = self.thresholds["min_precision"]
            alerts.append(f"precision {metrics['precision']:.4f} < {min_prec}")
        if metrics["f1"] < self.thresholds["min_f1"]:
            alerts.append(f"f1 {metrics['f1']:.4f} < {self.thresholds['min_f1']}")

        metrics["alerts"] = alerts
        metrics["degraded"] = len(alerts) > 0
        return metrics


def run_performance_monitoring():
    params = load_params()
    root = get_project_root()

    eval_metrics_path = root / "metrics" / "eval_metrics.json"
    if not eval_metrics_path.exists():
        logger.warning("No evaluation metrics found — run evaluate first")
        return

    with open(eval_metrics_path) as f:
        eval_metrics = json.load(f)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "eval_metrics": eval_metrics,
        "thresholds": params["thresholds"],
        "degraded": False,
        "alerts": [],
    }

    for metric_name in ["recall", "precision", "f1", "auc_roc"]:
        threshold_key = f"min_{metric_name}"
        if eval_metrics.get(metric_name, 1.0) < params["thresholds"][threshold_key]:
            report["degraded"] = True
            report["alerts"].append(
                f"{metric_name}: {eval_metrics[metric_name]:.4f}"
                f" < {params['thresholds'][threshold_key]}"
            )

    reports_dir = root / "metrics" / "performance"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"perf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if report["degraded"]:
        logger.warning(f"MODEL DEGRADED: {report['alerts']}")
        _publish_degradation_alarm(report, params)
    else:
        logger.info("Model performance is within acceptable thresholds")

    return report


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


if __name__ == "__main__":
    run_performance_monitoring()
