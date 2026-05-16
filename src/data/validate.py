"""Data validation: schema checks and statistical tests on ingested data."""

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

from src.utils.config import load_params, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_schema() -> DataFrameSchema:
    v_columns = {
        f"V{i}": Column(float, nullable=False) for i in range(1, 29)
    }
    return DataFrameSchema(
        columns={
            "Time": Column(float, Check.ge(0), nullable=False),
            **v_columns,
            "Amount": Column(float, Check.ge(0), nullable=False),
            "Class": Column(float, Check.isin([0.0, 1.0]), nullable=False),
        },
        coerce=True,
    )


def validate_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    params = load_params()
    if df is None:
        raw_path = get_project_root() / params["data"]["raw_path"]
        df = pd.read_csv(raw_path)

    schema = get_schema()
    validated_df = schema.validate(df, lazy=True)
    logger.info(f"Schema validation passed — {len(validated_df)} rows")

    null_counts = validated_df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found")

    dup_count = validated_df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    fraud_ratio = validated_df["Class"].mean()
    logger.info(f"Fraud ratio: {fraud_ratio:.4f}")
    if fraud_ratio > 0.5:
        raise ValueError(f"Unexpected fraud ratio {fraud_ratio:.4f} — data may be corrupted")

    return validated_df


if __name__ == "__main__":
    validate_data()
