"""
tests/test_prediction.py
========================
Tests for the PredictionEngine (Option-A live-rating path).

All Chess.com calls are mocked so the suite runs fully offline.
"""

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chess_api import ChessAPIClient
from src.prediction import PredictionEngine

EXPECTED_KEYS = {
    "success",
    "error",
    "probabilities",
    "predicted_class",
    "player_a",
    "player_b",
    "features_used",
    "shap_explanation",
    "warnings",
}

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_model.pkl"
)

# Distinct ratings so A/B <-> white/black mapping is observable
RATING_A = 1600
RATING_B = 1400


def _fake_rating(username, time_class):
    """Return a fixed rating per username without touching the network."""
    if username == "player_a":
        return RATING_A
    if username == "player_b":
        return RATING_B
    return None


class TestPredictionEngine:
    """Tests for PredictionEngine."""

    def setup_method(self):
        self.engine = PredictionEngine()

    # ------------------------------------------------------------------

    def test_prediction_no_model(self):
        """When no model file exists, the result must have success=False and a descriptive error."""
        result = self.engine.predict(
            username_a="player_a",
            username_b="player_b",
            color_a="white",
            time_class="blitz",
            rated=True,
            model_path="/nonexistent/path/model.pkl",
        )
        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0

    def test_prediction_structure(self):
        """The result dict must always contain all expected keys."""
        result = self.engine.predict(
            username_a="player_a",
            username_b="player_b",
            color_a="white",
            time_class="blitz",
            rated=True,
            model_path="/nonexistent/path/model.pkl",
        )
        for key in EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_prediction_probabilities_sum_to_one(self):
        """A successful prediction must have probabilities summing to ~100 %."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No trained model available")

        with patch.object(
            ChessAPIClient, "get_player_rating", side_effect=_fake_rating
        ):
            result = self.engine.predict(
                username_a="player_a",
                username_b="player_b",
                color_a="white",
                time_class="blitz",
                rated=True,
                model_path=MODEL_PATH,
            )

        assert result["success"] is True, result["error"]
        probs = result["probabilities"]
        total = probs["win_a"] + probs["win_b"] + probs["draw"]
        assert abs(total - 100.0) < 0.1, f"Probabilities sum to {total}, not 100"

    def test_mapping_color_a_white(self):
        """color_a='white' -> A is white, white_rating=rating_a, black_rating=rating_b."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No trained model available")

        with patch.object(
            ChessAPIClient, "get_player_rating", side_effect=_fake_rating
        ):
            result = self.engine.predict(
                username_a="player_a",
                username_b="player_b",
                color_a="white",
                time_class="blitz",
                rated=True,
                model_path=MODEL_PATH,
            )

        assert result["success"] is True, result["error"]
        assert result["player_a"]["color"] == "white"
        assert result["player_b"]["color"] == "black"
        assert result["player_a"]["rating"] == RATING_A
        assert result["player_b"]["rating"] == RATING_B
        feats = result["features_used"]
        assert feats["white_rating"] == RATING_A
        assert feats["black_rating"] == RATING_B

    def test_mapping_color_a_black(self):
        """color_a='black' -> A is black, white_rating=rating_b, black_rating=rating_a."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No trained model available")

        with patch.object(
            ChessAPIClient, "get_player_rating", side_effect=_fake_rating
        ):
            result = self.engine.predict(
                username_a="player_a",
                username_b="player_b",
                color_a="black",
                time_class="blitz",
                rated=True,
                model_path=MODEL_PATH,
            )

        assert result["success"] is True, result["error"]
        assert result["player_a"]["color"] == "black"
        assert result["player_b"]["color"] == "white"
        feats = result["features_used"]
        # White is player B, black is player A
        assert feats["white_rating"] == RATING_B
        assert feats["black_rating"] == RATING_A
