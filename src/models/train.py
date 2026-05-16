"""Model training with MLflow experiment tracking and XGBoost."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.utils.config import load_params, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }


def train_model() -> Path:
    params = load_params()
    root = get_project_root()

    processed_dir = root / params["data"]["processed_path"]
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    X_val = pd.read_csv(processed_dir / "X_val.csv")
    y_val = pd.read_csv(processed_dir / "y_val.csv").squeeze()

    model_params = params["model"]["params"]

    mlflow.set_tracking_uri(str(root / "mlruns"))
    mlflow.set_experiment(params["training"]["experiment_name"])

    with mlflow.start_run(run_name="xgboost-fraud-detection") as run:
        mlflow.log_params(model_params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("train_fraud_ratio", float(y_train.mean()))
        mlflow.log_param("n_features", X_train.shape[1])

        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)
        logger.info(f"Validation metrics: {val_metrics}")

        logger.info("Running cross-validation...")
        cv = StratifiedKFold(
            n_splits=params["training"]["cv_folds"],
            shuffle=True,
            random_state=params["data"]["random_state"],
        )
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            fold_model = xgb.XGBClassifier(**model_params)
            fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
            fold_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
            fold_auc = roc_auc_score(y_train.iloc[val_idx], fold_prob)
            cv_scores.append(fold_auc)
            mlflow.log_metric("cv_auc_roc", fold_auc, step=fold)

        mlflow.log_metric("cv_auc_roc_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_auc_roc_std", float(np.std(cv_scores)))
        logger.info(f"CV AUC-ROC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        feature_importance = dict(
            zip(X_train.columns, model.feature_importances_.tolist())
        )
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top features: {top_features}")

        mlflow.xgboost.log_model(model, "model", input_example=X_val.head(1))

        model_dir = root / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        metrics_dir = root / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        train_metrics = {
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "cv_auc_roc_mean": float(np.mean(cv_scores)),
            "cv_auc_roc_std": float(np.std(cv_scores)),
            "run_id": run.info.run_id,
        }
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"MLflow run ID: {run.info.run_id}")

    return model_path


if __name__ == "__main__":
    train_model()
