"""Feature engineering and train/val/test splitting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.validate import validate_data
from src.utils.config import load_params, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data() -> dict[str, Path]:
    params = load_params()
    root = get_project_root()

    raw_path = root / params["data"]["raw_path"]
    processed_dir = root / params["data"]["processed_path"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = validate_data(df)

    drop_cols = params["features"]["drop_columns"]
    target_col = params["features"]["target_column"]

    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    relative_val_size = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        random_state=params["data"]["random_state"],
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    scale_cols = ["Amount"]
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    output_files = {}
    for name, data in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        path = processed_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        output_files[name] = path

    scaler_path = processed_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    output_files["scaler"] = scaler_path

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
    logger.info(f"Scaler saved to {scaler_path}")

    return output_files


if __name__ == "__main__":
    preprocess_data()
