import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.unit
class TestModel:
    def _train_dummy_model(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10))
        y = rng.choice([0, 1], 500, p=[0.95, 0.05])
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y, verbose=False)
        return model

    def test_model_outputs_probabilities(self):
        model = self._train_dummy_model()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((10, 10))
        probs = model.predict_proba(X)
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_model_outputs_binary_predictions(self):
        model = self._train_dummy_model()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((10, 10))
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_deterministic(self):
        model = self._train_dummy_model()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((10, 10))
        pred1 = model.predict_proba(X)
        pred2 = model.predict_proba(X)
        np.testing.assert_array_equal(pred1, pred2)

    def test_model_handles_single_sample(self):
        model = self._train_dummy_model()
        X = np.random.default_rng(99).standard_normal((1, 10))
        pred = model.predict(X)
        assert len(pred) == 1
