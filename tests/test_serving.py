"""
tests/test_serving.py
=====================
Tests that the deployed model artifact exposes a uniform,
architecture-agnostic interface (works for both the single multiclass
model and the two-stage wrapper).

All tests run offline; no network access.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib

from src.feature_engineering import FeatureEngineer

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_model.pkl"
)


def _hand_built_row() -> pd.DataFrame:
    """Return a single valid Option-A feature row as a DataFrame."""
    fe = FeatureEngineer()
    series = fe.build_match_features(
        white_rating=1650,
        black_rating=1500,
        time_class="blitz",
        rated=True,
    )
    return pd.DataFrame([series], columns=fe.get_feature_names())


class TestDeployedArtifact:
    """The deployed model must expose a uniform interface regardless of arch."""

    def setup_method(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No trained model available")
        self.model = joblib.load(MODEL_PATH)

    def test_uniform_interface(self):
        """Deployed model exposes predict, predict_proba and feature_names."""
        assert hasattr(self.model, "predict")
        assert hasattr(self.model, "predict_proba")
        assert hasattr(self.model, "feature_names")
        assert list(self.model.feature_names) == FeatureEngineer.FEATURE_NAMES

    def test_predict_proba_shape_and_sum(self):
        """predict_proba returns an (n, 3) array whose rows sum to ~1.0."""
        X = _hand_built_row()
        proba = self.model.predict_proba(X)
        assert proba.shape == (1, 3)
        assert abs(float(proba.sum(axis=1)[0]) - 1.0) < 1e-6

    def test_predict_returns_valid_class(self):
        """predict returns a label in {0, 1, 2}."""
        X = _hand_built_row()
        pred = self.model.predict(X)
        assert pred.shape == (1,)
        assert int(pred[0]) in (0, 1, 2)

    def test_batch_rows_sum_to_one(self):
        """A multi-row batch also yields per-row probabilities summing to 1."""
        X = pd.concat([_hand_built_row(), _hand_built_row()], ignore_index=True)
        proba = self.model.predict_proba(X)
        assert proba.shape == (2, 3)
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)
