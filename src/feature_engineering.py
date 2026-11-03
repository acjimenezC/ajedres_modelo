"""
feature_engineering.py
=======================
Builds the Option-A per-match feature vector for the chess-result model.

Option A predicts a single match from the per-match context only (the two
players' current ratings, the rating gap, and the time-control setup).
Recent-history features are impossible with this dataset because ~86% of
users appear in only one game, so every feature is derived from the match
itself with no reliance on prior games.
"""

from typing import List

import numpy as np
import pandas as pd

# Target class constants
TARGET_WIN_WHITE = 0
TARGET_WIN_BLACK = 1
TARGET_DRAW = 2

# Time-class encoding
TIME_CLASS_MAP = {"bullet": 0, "blitz": 1, "rapid": 2, "daily": 3}


class FeatureEngineer:
    """
    Builds the fixed 6-feature Option-A vector for a single match.
    """

    FEATURE_NAMES: List[str] = [
        "white_rating",
        "black_rating",
        "rating_diff",
        "abs_rating_diff",
        "time_class_encoded",
        "rated",
    ]

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        return list(self.FEATURE_NAMES)

    def build_match_features(
        self,
        white_rating,
        black_rating,
        time_class: str,
        rated,
    ) -> pd.Series:
        """
        Build the Option-A feature vector for a single match.

        Parameters
        ----------
        white_rating : int or float
            ELO rating of the white player.
        black_rating : int or float
            ELO rating of the black player.
        time_class : str
            One of 'bullet', 'blitz', 'rapid', 'daily'.
        rated : bool or int
            Whether the game was rated (ELO points at stake).

        Returns
        -------
        pd.Series
            Feature vector indexed by FEATURE_NAMES.
        """
        rating_diff = white_rating - black_rating
        abs_rating_diff = abs(rating_diff)
        time_class_encoded = TIME_CLASS_MAP.get(str(time_class).lower(), 1)

        values = [
            white_rating,
            black_rating,
            rating_diff,
            abs_rating_diff,
            time_class_encoded,
            int(rated),
        ]

        return pd.Series(values, index=self.FEATURE_NAMES)
