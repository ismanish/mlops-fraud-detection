"""Model evaluation against holdout test set with quality gates."""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import get_project_root, load_params
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model() -> dict:
    params = load_params()
    root = get_project_root()

    model = joblib.load(root / "models" / "model.pkl")

    processed_dir = root / params["data"]["processed_path"]
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "test_samples": len(y_test),
        "fraud_count": int(y_test.sum()),
        "predicted_fraud_count": int(y_pred.sum()),
    }

    logger.info(f"Test metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    metrics_dir = root / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    with open(metrics_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    cm_data = {
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1]),
    }
    with open(metrics_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = [
        {"fpr": float(fpr[i]), "tpr": float(tpr[i])}
        for i in range(0, len(fpr), max(1, len(fpr) // 100))
    ]
    with open(metrics_dir / "roc_curve.json", "w") as f:
        json.dump(roc_data, f, indent=2)

    _check_quality_gates(metrics, params)

    return metrics


def _check_quality_gates(metrics: dict, params: dict) -> None:
    thresholds = params["thresholds"]
    gates = {
        "recall": ("min_recall", metrics["recall"]),
        "precision": ("min_precision", metrics["precision"]),
        "f1": ("min_f1", metrics["f1"]),
        "auc_roc": ("min_auc_roc", metrics["auc_roc"]),
    }

    failures = []
    for metric_name, (threshold_key, actual) in gates.items():
        minimum = thresholds[threshold_key]
        status = "PASS" if actual >= minimum else "FAIL"
        logger.info(f"  Quality gate {metric_name}: {actual:.4f} >= {minimum} → {status}")
        if actual < minimum:
            failures.append(f"{metric_name}: {actual:.4f} < {minimum}")

    if failures:
        msg = "Quality gates FAILED: " + "; ".join(failures)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("All quality gates PASSED")


if __name__ == "__main__":
    evaluate_model()
