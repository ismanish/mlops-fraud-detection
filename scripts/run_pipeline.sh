#!/bin/bash
set -euo pipefail

echo "=== Running Full ML Pipeline ==="

echo "Step 1: Data Ingestion"
python -m src.data.ingest

echo "Step 2: Data Validation"
python -m src.data.validate

echo "Step 3: Data Preprocessing"
python -m src.data.preprocess

echo "Step 4: Model Training"
python -m src.models.train

echo "Step 5: Model Evaluation"
python -m src.models.evaluate

echo "Step 6: Drift Detection"
python -m src.monitoring.drift_detection

echo "Step 7: Performance Monitoring"
python -m src.monitoring.performance

echo ""
echo "=== Pipeline Complete ==="
echo "View MLflow UI: mlflow ui --port 5000"
echo "View metrics: cat metrics/eval_metrics.json"
