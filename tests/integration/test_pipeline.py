import pandas as pd
import pytest

from src.data.ingest import generate_synthetic_fraud_data
from src.data.validate import validate_data


@pytest.mark.integration
class TestPipeline:
    def test_ingest_generates_correct_shape(self):
        df = generate_synthetic_fraud_data(n_samples=1000)
        assert df.shape[0] == 1000
        assert df.shape[1] == 31  # Time + V1-V28 + Amount + Class

    def test_ingest_generates_fraud_and_legit(self):
        df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.01)
        assert df["Class"].sum() > 0
        assert (df["Class"] == 0).sum() > 0

    def test_validate_passes_on_clean_data(self):
        df = generate_synthetic_fraud_data(n_samples=1000)
        validated = validate_data(df)
        assert len(validated) == 1000

    def test_validate_rejects_bad_data(self):
        df = generate_synthetic_fraud_data(n_samples=100)
        df.loc[0, "Class"] = 5.0  # invalid class
        with pytest.raises(Exception):
            validate_data(df)
