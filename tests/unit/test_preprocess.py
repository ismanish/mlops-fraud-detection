import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.mark.unit
class TestPreprocessing:
    def _make_sample_data(self, n: int = 1000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        data = {
            "Time": rng.uniform(0, 172800, n),
            "Amount": rng.exponential(100, n),
            "Class": rng.choice([0, 1], n, p=[0.99, 0.01]),
        }
        for i in range(1, 29):
            data[f"V{i}"] = rng.standard_normal(n)
        return pd.DataFrame(data)

    def test_no_nulls_after_preprocessing(self):
        df = self._make_sample_data()
        df = df.drop(columns=["Time"])
        assert df.isnull().sum().sum() == 0

    def test_class_column_is_binary(self):
        df = self._make_sample_data()
        assert set(df["Class"].unique()).issubset({0, 1})

    def test_scaling_standardizes_amount(self):
        df = self._make_sample_data()
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
        assert abs(df["Amount"].mean()) < 0.1
        assert abs(df["Amount"].std() - 1.0) < 0.1

    def test_train_test_split_preserves_ratio(self):
        from sklearn.model_selection import train_test_split

        df = self._make_sample_data(n=10000)
        y = df["Class"]
        _, _, y_train, y_test = train_test_split(
            df, y, test_size=0.2, stratify=y, random_state=42
        )
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.01

    def test_feature_count_after_drop(self):
        df = self._make_sample_data()
        df = df.drop(columns=["Time", "Class"])
        assert df.shape[1] == 29  # V1-V28 + Amount
