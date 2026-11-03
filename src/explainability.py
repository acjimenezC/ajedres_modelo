"""
explainability.py
=================
SHAP-based model interpretability.

Selects the appropriate SHAP explainer based on the model type and
produces summary, importance plots, and per-prediction explanations.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelExplainer:
    """
    Generates SHAP explanations for chess match outcome predictions.

    Chooses the explainer automatically based on the classifier type:
    - TreeExplainer  : RandomForest, GradientBoosting, DecisionTree
    - LinearExplainer: LogisticRegression
    - KernelExplainer: SVM, KNN (model-agnostic; uses a background sample)
    """

    # ------------------------------------------------------------------
    # SHAP value computation
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        model,
        X: pd.DataFrame,
        feature_names: List[str],
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for the given model and data.

        Parameters
        ----------
        model : fitted sklearn estimator (or Pipeline)
        X : pd.DataFrame
        feature_names : list of str

        Returns
        -------
        np.ndarray of shape (n_samples, n_features, n_classes) or
        (n_samples, n_features) for binary.
        Returns None if SHAP is not available or computation fails.
        """
        if not SHAP_AVAILABLE:
            return None

        # If it's a Pipeline, extract the classifier and transform X
        clf, X_transformed = self._extract_clf_and_transform(model, X)

        # Convert to numpy array with feature names preserved
        if hasattr(X_transformed, "toarray"):
            X_arr = X_transformed.toarray()
        else:
            X_arr = np.array(X_transformed)

        explainer = self._get_explainer(clf, X_arr)
        if explainer is None:
            return None

        try:
            shap_vals = explainer.shap_values(X_arr)
            return shap_vals
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None

    def _extract_clf_and_transform(self, model, X):
        """
        Extract the underlying classifier from a model and transform X.

        For a TwoStageModel the explanation targets stage 2 (white vs black),
        the stage that carries the colour decision; X is pushed through that
        stage's preprocessing steps before SHAP sees the XGBoost booster.
        """
        from sklearn.pipeline import Pipeline as SKPipeline

        # TwoStageModel: explain the colour stage (stage 2)
        if hasattr(model, "stage2") and hasattr(model, "stage1"):
            return self._extract_clf_and_transform(model.stage2, X)

        # MulticlassModel adapter: explain the wrapped multiclass pipeline
        if hasattr(model, "pipeline") and not isinstance(model, SKPipeline):
            return self._extract_clf_and_transform(model.pipeline, X)

        if isinstance(model, SKPipeline):
            # Transform X through all steps except the last
            X_t = X.copy()
            steps = model.steps[:-1]
            for _, transformer in steps:
                X_t = transformer.transform(X_t)
            clf = model.steps[-1][1]
        else:
            X_t = X
            clf = model
        return clf, X_t

    def _get_explainer(self, clf, X_arr):
        """Return the appropriate SHAP explainer for clf."""
        clf_type = type(clf).__name__

        try:
            if clf_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                            "DecisionTreeClassifier", "XGBClassifier"):
                return shap.TreeExplainer(clf)
            elif clf_type == "LogisticRegression":
                return shap.LinearExplainer(clf, X_arr)
            else:
                # KernelExplainer for SVM, KNN, etc.
                background = shap.sample(X_arr, min(50, len(X_arr)))
                return shap.KernelExplainer(clf.predict_proba, background)
        except Exception as e:
            print(f"Could not create SHAP explainer: {e}")
            return None

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_shap_summary(
        self,
        shap_values,
        X: pd.DataFrame,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        SHAP beeswarm summary plot.

        For multiclass, uses the first class by default (win_a).
        """
        if not SHAP_AVAILABLE or shap_values is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.title("SHAP — Resumen de Importancia de Variables", fontsize=13)

        # For multiclass shap_values is a list of arrays (one per class)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        X_disp = pd.DataFrame(
            np.array(X)[:, :len(feature_names)],
            columns=feature_names,
        )

        shap.summary_plot(
            vals,
            X_disp,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=120)
            plt.close()

        return fig

    def plot_shap_importance(
        self,
        shap_values,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Bar chart of mean absolute SHAP values (global feature importance).
        """
        if not SHAP_AVAILABLE or shap_values is None:
            return None

        # For multiclass, average across classes
        if isinstance(shap_values, list):
            mean_abs = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        # Align lengths
        n = min(len(mean_abs), len(feature_names))
        importance_df = pd.DataFrame(
            {"Feature": feature_names[:n], "SHAP": mean_abs[:n]}
        ).sort_values("SHAP", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(5, n // 2)))
        ax.barh(importance_df["Feature"], importance_df["SHAP"], color="steelblue")
        ax.set_title("Importancia de Variables (SHAP)", fontsize=13)
        ax.set_xlabel("Valor SHAP medio absoluto")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)

        return fig

    # ------------------------------------------------------------------
    # Per-prediction explanation
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        model,
        X_instance: pd.DataFrame,
        feature_names: List[str],
    ) -> dict:
        """
        Return the top 5 most influential features for a single prediction.

        Parameters
        ----------
        model : fitted pipeline
        X_instance : pd.DataFrame (single row)
        feature_names : list of str

        Returns
        -------
        dict with keys:
            top_features : list of (feature_name, shap_value, direction)
            raw_shap_values : list of floats
        """
        if not SHAP_AVAILABLE:
            return {
                "top_features": [],
                "raw_shap_values": [],
                "error": "SHAP not installed",
            }

        shap_vals = self.compute_shap_values(model, X_instance, feature_names)
        if shap_vals is None:
            return {"top_features": [], "raw_shap_values": [], "error": "SHAP failed"}

        # Normalise SHAP output to a 1-D per-feature vector for the first
        # (only) instance.  The shape depends on the explainer:
        #   - list of (n, f) arrays  : one entry per class (older SHAP API)
        #   - ndarray (n, f, c)      : multiclass XGBoost (f features, c classes)
        #   - ndarray (n, f)         : binary / single-output
        # For the two-stage model the colour stage (stage 2) drives the
        # explanation; for the multiclass model the first class is used.
        if isinstance(shap_vals, list):
            vals = np.asarray(shap_vals[0])[0]  # first class, first instance
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 3:
                vals = arr[0, :, 0]  # first instance, all features, first class
            else:
                vals = arr[0]
        vals = np.asarray(vals).ravel()

        n = min(len(vals), len(feature_names))
        paired = list(zip(feature_names[:n], vals[:n]))
        sorted_pairs = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
        top5 = sorted_pairs[:5]

        top_features = [
            {
                "feature": name,
                "shap_value": float(val),
                "direction": "positive" if val > 0 else "negative",
            }
            for name, val in top5
        ]

        return {
            "top_features": top_features,
            "raw_shap_values": [float(v) for v in vals[:n]],
        }
