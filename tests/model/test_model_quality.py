import json
from pathlib import Path

import pytest

from src.utils.config import get_project_root


@pytest.mark.model
class TestModelQuality:
    def _load_metrics(self) -> dict:
        path = get_project_root() / "metrics" / "eval_metrics.json"
        if not path.exists():
            pytest.skip("eval_metrics.json not found — run evaluate first")
        with open(path) as f:
            return json.load(f)

    def test_auc_roc_above_threshold(self):
        metrics = self._load_metrics()
        assert metrics["auc_roc"] >= 0.90, f"AUC-ROC {metrics['auc_roc']} below 0.90"

    def test_recall_above_threshold(self):
        metrics = self._load_metrics()
        assert metrics["recall"] >= 0.70, f"Recall {metrics['recall']} below 0.70"

    def test_precision_above_threshold(self):
        metrics = self._load_metrics()
        assert metrics["precision"] >= 0.30, f"Precision {metrics['precision']} below 0.30"

    def test_model_not_predicting_all_same_class(self):
        metrics = self._load_metrics()
        assert metrics["predicted_fraud_count"] > 0, "Model predicts no fraud"
        assert metrics["predicted_fraud_count"] < metrics["test_samples"], "Model predicts all fraud"
