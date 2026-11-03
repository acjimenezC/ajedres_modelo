"""
prediction.py
=============
High-level prediction engine (Option-A live-rating path).

Orchestrates the full prediction pipeline:
  1. Fetch each player's CURRENT rating from Chess.com (/stats endpoint)
  2. Map players A/B to white/black from the chosen color of player A
  3. Build the Option-A per-match feature vector
  4. Load the deployed model and return class probabilities
  5. Attach SHAP explanations when available

The deployed model is architecture-agnostic: it may be the single
multiclass XGBoost (MulticlassModel) or the two-stage wrapper
(TwoStageModel).  Both expose the same interface — predict_proba(X)
returning columns [win_white, win_black, draw], predict(X) and
feature_names — so this engine does not care which architecture won.

IMPORTANT: This module NEVER returns invented probabilities.
  If the API fails or the model is missing, success=False is returned.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# Allow running as script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chess_api import ChessAPIClient
from src.explainability import ModelExplainer
from src.feature_engineering import FeatureEngineer

CLASS_LABELS = {0: "win_a", 1: "win_b", 2: "draw"}


class PredictionEngine:
    """
    Orchestrates end-to-end prediction for a single chess match using the
    Option-A per-match feature set and live current ratings from Chess.com.
    """

    def __init__(self):
        self.api = ChessAPIClient()
        self.fe = FeatureEngineer()
        self.explainer = ModelExplainer()

    def predict(
        self,
        username_a: str,
        username_b: str,
        color_a: str,
        time_class: str,
        rated,
        model_path: str = "models/best_model.pkl",
    ) -> dict:
        """
        Predict the outcome of a chess match between two players.

        The model works in white/black terms, while the UI speaks
        in terms of player A and player B. The A/B <-> white/black mapping is
        driven by ``color_a``:

        * color_a == 'white' -> A is white, B is black
            white_rating = rating_a, black_rating = rating_b
            win_white -> win_a, win_black -> win_b
        * color_a == 'black' -> A is black, B is white
            white_rating = rating_b, black_rating = rating_a
            win_white -> win_b, win_black -> win_a

        Draw maps to draw in both directions.

        Parameters
        ----------
        username_a, username_b : str
            Chess.com usernames.
        color_a : str
            Color assigned to player A ('white' or 'black').
        time_class : str
            One of 'bullet', 'blitz', 'rapid', 'daily'.
        rated : bool or int
            Whether the game is rated (ELO points at stake).
        model_path : str
            Path to the persisted deployed model (multiclass or two-stage).

        Returns
        -------
        dict — see module docstring for full structure
        """
        result = {
            "success": False,
            "error": None,
            "probabilities": None,
            "predicted_class": None,
            "player_a": None,
            "player_b": None,
            "features_used": None,
            "shap_explanation": None,
            "warnings": [],
        }

        # ----------------------------------------------------------------
        # 1. Load model
        # ----------------------------------------------------------------
        # Resolve path relative to project root if not absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__), "..", model_path
            )
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            result["error"] = (
                f"Modelo no encontrado en '{model_path}'. "
                "Ejecuta 'python -m src.data_loader' y luego "
                "'python -m src.train' para generar el modelo."
            )
            return result

        try:
            model = joblib.load(model_path)
        except Exception as e:
            result["error"] = f"Error al cargar el modelo: {e}"
            return result

        # ----------------------------------------------------------------
        # 2. Fetch current ratings from Chess.com
        # ----------------------------------------------------------------
        rating_a = self.api.get_player_rating(username_a, time_class)
        if rating_a is None:
            result["error"] = (
                f"No se pudo obtener el rating de '{username_a}' en "
                f"{time_class}. Verifica el nombre de usuario y que tenga "
                "partidas en esa modalidad."
            )
            return result

        rating_b = self.api.get_player_rating(username_b, time_class)
        if rating_b is None:
            result["error"] = (
                f"No se pudo obtener el rating de '{username_b}' en "
                f"{time_class}. Verifica el nombre de usuario y que tenga "
                "partidas en esa modalidad."
            )
            return result

        # ----------------------------------------------------------------
        # 3. Map A/B to white/black from color_a
        # ----------------------------------------------------------------
        color_a = str(color_a).lower()
        if color_a == "white":
            color_b = "black"
            white_rating, black_rating = rating_a, rating_b
        else:
            color_a = "black"
            color_b = "white"
            white_rating, black_rating = rating_b, rating_a

        # ----------------------------------------------------------------
        # 4. Build the Option-A feature vector
        # ----------------------------------------------------------------
        feature_series = self.fe.build_match_features(
            white_rating,
            black_rating,
            time_class,
            rated,
        )
        feature_names = self.fe.get_feature_names()
        X = pd.DataFrame([feature_series], columns=feature_names)

        # ----------------------------------------------------------------
        # 5. Predict (model columns are [win_white, win_black, draw])
        # ----------------------------------------------------------------
        try:
            proba = model.predict_proba(X)[0]
        except Exception as e:
            result["error"] = f"Error durante la predicción: {e}"
            return result

        # Verify probs sum to 1
        if not np.isclose(proba.sum(), 1.0, atol=1e-6):
            proba = proba / proba.sum()

        p_white = float(proba[0])
        p_black = float(proba[1])
        p_draw = float(proba[2])

        # Translate white/black back to A/B using the same color mapping
        if color_a == "white":
            p_win_a, p_win_b = p_white, p_black
        else:
            p_win_a, p_win_b = p_black, p_white

        probabilities = {
            "win_a": round(p_win_a * 100, 2),
            "win_b": round(p_win_b * 100, 2),
            "draw": round(p_draw * 100, 2),
        }

        pred_idx = int(np.argmax([p_win_a, p_win_b, p_draw]))
        pred_class_label = CLASS_LABELS[pred_idx]

        # ----------------------------------------------------------------
        # 6. Player summaries
        # ----------------------------------------------------------------
        result["player_a"] = {
            "username": username_a,
            "rating": rating_a,
            "color": color_a,
        }
        result["player_b"] = {
            "username": username_b,
            "rating": rating_b,
            "color": color_b,
        }

        # ----------------------------------------------------------------
        # 7. SHAP explanation (optional; graceful failure)
        # ----------------------------------------------------------------
        try:
            shap_exp = self.explainer.explain_prediction(model, X, feature_names)
        except Exception:
            shap_exp = None
        result["shap_explanation"] = shap_exp

        # ----------------------------------------------------------------
        # 8. Compose result
        # ----------------------------------------------------------------
        result.update(
            {
                "success": True,
                "probabilities": probabilities,
                "predicted_class": pred_class_label,
                "features_used": feature_series.to_dict(),
            }
        )
        return result
