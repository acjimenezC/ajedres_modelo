"""
tests/test_preprocessing.py
============================
Tests for game result parsing, the Option-A feature set, preprocessing, and
the time_control parser.
"""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import DataPreprocessor, DRAW_RESULTS, LOSS_RESULTS
from src.feature_engineering import FeatureEngineer
from src.data_loader import parse_time_control


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    def setup_method(self):
        self.pre = DataPreprocessor()

    # ------------------------------------------------------------------
    # parse_game_result
    # ------------------------------------------------------------------

    def test_parse_result_win(self):
        """A game where result='win' should map to 'win'."""
        game = {"result": "win", "color": "white"}
        assert self.pre.parse_game_result(game, "player1") == "win"

    def test_parse_result_draw(self):
        """All draw codes should map to 'draw'."""
        for draw_code in DRAW_RESULTS:
            game = {"result": draw_code, "color": "white"}
            assert self.pre.parse_game_result(game, "player1") == "draw", \
                f"Expected 'draw' for result={draw_code}"

    def test_parse_result_loss(self):
        """All loss codes should map to 'loss'."""
        for loss_code in LOSS_RESULTS:
            game = {"result": loss_code, "color": "black"}
            assert self.pre.parse_game_result(game, "player1") == "loss", \
                f"Expected 'loss' for result={loss_code}"

    # ------------------------------------------------------------------
    # clean_games
    # ------------------------------------------------------------------

    def test_clean_games_removes_abandoned(self):
        """Games with result='abandoned' should be removed by clean_games."""
        games = [
            {"result": "win"},
            {"result": "abandoned"},
            {"result": "checkmated"},
        ]
        cleaned = self.pre.clean_games(games)
        results = [g["result"] for g in cleaned]
        assert "abandoned" not in results
        assert "win" in results

    # ------------------------------------------------------------------
    # encode_result_for_target
    # ------------------------------------------------------------------

    def test_result_encoding(self):
        """encode_result_for_target must return correct integer class labels."""
        assert self.pre.encode_result_for_target("win") == 0
        assert self.pre.encode_result_for_target("loss") == 1
        assert self.pre.encode_result_for_target("draw") == 2
        assert self.pre.encode_result_for_target("unknown") == -1

    # ------------------------------------------------------------------
    # prepare_features (Option-A)
    # ------------------------------------------------------------------

    def test_prepare_features_option_a(self):
        """prepare_features returns X with exactly the 6 FEATURE_NAMES and a 3-class y."""
        rows = [
            [1500, 1400, 100, 100, 1, 1, 0],
            [1200, 1300, -100, 100, 0, 1, 1],
            [1600, 1600, 0, 0, 2, 0, 2],
            [1000, 1100, -100, 100, 1, 1, -1],  # dropped (target == -1)
        ]
        cols = FeatureEngineer.FEATURE_NAMES + ["target"]
        df = pd.DataFrame(rows, columns=cols)

        X, y = self.pre.prepare_features(df)

        assert list(X.columns) == FeatureEngineer.FEATURE_NAMES
        assert len(X.columns) == 6
        assert len(X) == 3  # target == -1 row dropped
        assert set(y.unique()) == {0, 1, 2}


class TestFeatureSet:
    """Tests for the Option-A feature vector and time_control parsing."""

    def test_feature_names_order_and_length(self):
        """build_match_features must produce the 6 features in the exact order."""
        expected = [
            "white_rating",
            "black_rating",
            "rating_diff",
            "abs_rating_diff",
            "time_class_encoded",
            "rated",
        ]
        fe = FeatureEngineer()
        assert fe.FEATURE_NAMES == expected
        assert fe.get_feature_names() == expected

        series = fe.build_match_features(1500, 1400, "blitz", True)
        assert list(series.index) == expected
        assert len(series) == 6
        assert series["rating_diff"] == 100
        assert series["abs_rating_diff"] == 100

    def test_parse_time_control(self):
        """parse_time_control must cover the three time-control formats."""
        assert parse_time_control("60") == (60, 0)
        assert parse_time_control("180+2") == (180, 2)
        assert parse_time_control("1/259200") == (259200, 0)
