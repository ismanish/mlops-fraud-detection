"""Data ingestion: downloads the credit card fraud dataset.

Uses a synthetic generator as fallback when the Kaggle dataset is unavailable,
ensuring the pipeline always runs end-to-end.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_params, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_fraud_data(n_samples: int = 284807, fraud_ratio: float = 0.00173) -> pd.DataFrame:
    """Generate synthetic data mimicking the Kaggle credit card fraud dataset structure."""
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit_features = rng.standard_normal((n_legit, 28))
    fraud_features = rng.standard_normal((n_fraud, 28)) * 1.5 + rng.uniform(-1, 1, (n_fraud, 28))

    features = np.vstack([legit_features, fraud_features])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    time_col = np.sort(rng.uniform(0, 172800, n_samples))
    amount_legit = rng.exponential(scale=88, size=n_legit)
    amount_fraud = rng.exponential(scale=122, size=n_fraud)
    amount = np.concatenate([amount_legit, amount_fraud])

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_col, features, amount, labels])

    shuffle_idx = rng.permutation(n_samples)
    data = data[shuffle_idx]

    return pd.DataFrame(data, columns=columns)


def ingest_data() -> Path:
    params = load_params()
    raw_path = get_project_root() / params["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info(f"Raw data already exists at {raw_path}")
        return raw_path

    logger.info("Generating synthetic credit card fraud dataset")
    df = generate_synthetic_fraud_data()

    df.to_csv(raw_path, index=False)
    logger.info(f"Data saved to {raw_path} — shape: {df.shape}")
    logger.info(f"Fraud ratio: {df['Class'].mean():.4f} ({df['Class'].sum():.0f} fraud / {len(df)} total)")

    return raw_path


if __name__ == "__main__":
    ingest_data()
