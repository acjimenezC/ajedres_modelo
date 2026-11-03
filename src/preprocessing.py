"""
preprocessing.py
================
Data cleaning, result parsing, and sklearn preprocessing pipeline.

Handles Chess.com result codes and maps them to win/loss/draw/unknown.
Builds a sklearn Pipeline that imputes and scales numeric features.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# Chess.com result code mappings
# -----------------------------------------------------------------------
DRAW_RESULTS = {
    "repetition",
    "stalemate",
    "insufficient",
    "agreed",
    "50move",
    "timevsinsufficient",
}

LOSS_RESULTS = {
    "resigned",
    "timeout",
    "checkmated",
    "abandoned",
}


class DataPreprocessor:
    """
    Cleans raw game lists and builds sklearn preprocessing pipelines.

    Result interpretation follows Chess.com semantics:
    - 'win'            → that colour won
    - DRAW_RESULTS     → draw
    - LOSS_RESULTS     → that colour lost
    """

    # ------------------------------------------------------------------
    # Game-level helpers
    # ------------------------------------------------------------------

    def parse_game_result(self, game: dict, username: str) -> str:
        """
        Map the raw Chess.com result code to win / loss / draw / unknown.

        Parameters
        ----------
        game : dict
            Parsed game dict (as returned by ChessAPIClient._parse_game).
        username : str
            The player whose perspective we care about.

        Returns
        -------
        str
            'win', 'loss', 'draw', or 'unknown'.
        """
        result_code = game.get("result", "unknown").lower()
        if result_code == "win":
            return "win"
        if result_code in DRAW_RESULTS:
            return "draw"
        if result_code in LOSS_RESULTS:
            return "loss"
        return "unknown"

    def clean_games(self, games: list) -> list:
        """
        Remove abandoned and unknown-result games from a list.

        Parameters
        ----------
        games : list
            Raw game dicts from ChessAPIClient.

        Returns
        -------
        list
            Cleaned game dicts.
        """
        cleaned = []
        for game in games:
            result_code = game.get("result", "unknown").lower()
            opp_result = ""
            # Reconstruct opponent result from white/black structure if present
            # For already-parsed dicts the keys may not exist; skip gracefully
            if result_code == "abandoned":
                continue
            if result_code == "unknown":
                continue
            cleaned.append(game)
        return cleaned

    def encode_result_for_target(self, result_a: str) -> int:
        """
        Convert a win/loss/draw label for player A into the target integer.

        0 = win_a, 1 = win_b, 2 = draw

        Parameters
        ----------
        result_a : str
            Result from player A's perspective: 'win', 'loss', or 'draw'.

        Returns
        -------
        int
            Target class integer.
        """
        mapping = {"win": 0, "loss": 1, "draw": 2}
        return mapping.get(result_a.lower(), -1)

    # ------------------------------------------------------------------
    # DataFrame-level helpers
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split a processed dataset into feature matrix X and target y.

        Selects only the Option-A per-match feature columns and drops rows
        where the target is invalid (-1).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'target' column plus the Option-A FEATURE_NAMES.

        Returns
        -------
        (X, y) tuple
        """
        from src.feature_engineering import FeatureEngineer

        df = df.copy()
        df = df[df["target"] != -1]
        y = df["target"].astype(int)
        X = df[FeatureEngineer.FEATURE_NAMES].copy()
        return X, y

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Build a sklearn Pipeline that imputes then scales numeric features.

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        # We handle all columns as numeric (categorical encoding is done
        # upstream in FeatureEngineer before reaching this pipeline).
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ],
            remainder="drop",
        )
        return preprocessor
