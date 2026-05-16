"""Prediction module: loads model and returns fraud probability."""

import joblib
import pandas as pd

from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_scaler = None


def load_artifacts(model_path: str | None = None, scaler_path: str | None = None):
    global _model, _scaler
    root = get_project_root()

    if model_path is None:
        model_path = str(root / "models" / "model.pkl")
    if scaler_path is None:
        scaler_path = str(root / "data" / "processed" / "scaler.pkl")

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded")


def predict(features: dict | pd.DataFrame) -> dict:
    if _model is None:
        load_artifacts()

    if isinstance(features, dict):
        features = pd.DataFrame([features])

    if "Amount" in features.columns and _scaler is not None:
        features = features.copy()
        features["Amount"] = _scaler.transform(features[["Amount"]])

    if "Time" in features.columns:
        features = features.drop(columns=["Time"])

    assert _model is not None
    probability = float(_model.predict_proba(features)[:, 1][0])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "label": "fraud" if prediction == 1 else "legitimate",
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if _model is None:
        load_artifacts()

    features = df.copy()
    if "Amount" in features.columns and _scaler is not None:
        features["Amount"] = _scaler.transform(features[["Amount"]])
    if "Time" in features.columns:
        features = features.drop(columns=["Time"])

    assert _model is not None
    probabilities = _model.predict_proba(features)[:, 1]
    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = (probabilities >= 0.5).astype(int)
    return result
