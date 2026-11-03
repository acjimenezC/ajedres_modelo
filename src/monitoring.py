"""
monitoring.py
=============
Model monitoring and drift detection utilities.

Computes Population Stability Index (PSI), tracks prediction performance
over time, and recommends retraining when quality degrades.

IMPORTANT: retrain_model NEVER auto-replaces the production model.
  It validates the new model first and returns a report; the caller
  decides whether to deploy.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Allow running as script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# PSI interpretation thresholds
PSI_NO_CHANGE = 0.1
PSI_MODERATE = 0.25


class ModelMonitor:
    """
    Monitors a deployed chess prediction model for drift and degradation.
    """

    # ------------------------------------------------------------------
    # Population Stability Index
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Compute the Population Stability Index between two distributions.

        PSI < 0.1  : no significant change
        PSI 0.1–0.25: moderate change, monitor closely
        PSI > 0.25 : severe change, consider retraining

        Parameters
        ----------
        reference_dist : array-like  (training data distribution)
        current_dist   : array-like  (current/production data distribution)
        bins           : int

        Returns
        -------
        float
        """
        ref = np.array(reference_dist, dtype=float)
        cur = np.array(current_dist, dtype=float)

        # Determine bin edges from the combined range
        min_val = min(ref.min(), cur.min())
        max_val = max(ref.max(), cur.max())
        bin_edges = np.linspace(min_val, max_val + 1e-9, bins + 1)

        ref_pct, _ = np.histogram(ref, bins=bin_edges)
        cur_pct, _ = np.histogram(cur, bins=bin_edges)

        # Avoid division by zero / log(0) — replace 0 with a tiny value
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct / ref_pct.sum())
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct / cur_pct.sum())

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def interpret_psi(self, psi: float) -> str:
        """Return a human-readable interpretation of a PSI value."""
        if psi < PSI_NO_CHANGE:
            return "Sin cambio significativo"
        elif psi < PSI_MODERATE:
            return "Cambio moderado — monitorizar de cerca"
        else:
            return "Cambio severo — considerar reentrenamiento"

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def track_performance(self, predictions_log: List[dict]) -> dict:
        """
        Aggregate prediction performance over a log of past predictions.

        Each element of predictions_log must have:
            'predicted_class': str ('win_a'/'win_b'/'draw')
            'actual_class'   : str
            'probabilities'  : dict {'win_a': float, 'win_b': float, 'draw': float}

        Returns
        -------
        dict with accuracy, n_predictions, class_distribution
        """
        if not predictions_log:
            return {"n_predictions": 0, "accuracy": None}

        correct = 0
        class_counts: Dict[str, int] = {}

        for entry in predictions_log:
            pred = entry.get("predicted_class")
            actual = entry.get("actual_class")
            if pred == actual:
                correct += 1
            class_counts[actual] = class_counts.get(actual, 0) + 1

        n = len(predictions_log)
        return {
            "n_predictions": n,
            "accuracy": correct / n,
            "class_distribution": class_counts,
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """
        Compute PSI for every numeric feature between reference and current.

        Parameters
        ----------
        reference_df : pd.DataFrame  (training features)
        current_df   : pd.DataFrame  (recent production features)

        Returns
        -------
        dict of {feature_name: {'psi': float, 'interpretation': str}}
        """
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        drift_report: Dict[str, dict] = {}

        for col in numeric_cols:
            if col not in current_df.columns:
                continue
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue
            psi = self.compute_psi(ref_vals, cur_vals)
            drift_report[col] = {
                "psi": round(psi, 4),
                "interpretation": self.interpret_psi(psi),
            }

        return drift_report

    # ------------------------------------------------------------------
    # Retraining recommendation
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        metrics_history: List[dict],
        threshold_accuracy: float = 0.55,
        threshold_logloss: float = 1.2,
    ) -> bool:
        """
        Recommend retraining based on recent metric history.

        Only recommends; does NOT trigger retraining automatically.

        Parameters
        ----------
        metrics_history : list of dict, each with 'accuracy' and/or 'log_loss'
        threshold_accuracy : minimum acceptable accuracy
        threshold_logloss  : maximum acceptable log-loss

        Returns
        -------
        bool  (True = recommend retraining)
        """
        if not metrics_history:
            return False

        recent = metrics_history[-3:]  # last 3 evaluations

        acc_values = [m["accuracy"] for m in recent if "accuracy" in m]
        ll_values = [m["log_loss"] for m in recent if "log_loss" in m]

        if acc_values and np.mean(acc_values) < threshold_accuracy:
            return True
        if ll_values and np.mean(ll_values) > threshold_logloss:
            return True
        return False

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def retrain_model(
        self,
        new_data_path: str,
        old_model_path: str,
        output_path: str,
    ) -> dict:
        """
        Train a new model on new_data_path and validate before replacing.

        Does NOT automatically replace the old model.
        Returns a report so the caller can decide.

        Parameters
        ----------
        new_data_path : str  (CSV with features + target column)
        old_model_path : str
        output_path : str  (where to save the new model if validated)

        Returns
        -------
        dict with 'success', 'old_metrics', 'new_metrics', 'improved'
        """
        import joblib
        import pandas as pd

        from src.evaluate import ModelEvaluator
        from src.preprocessing import DataPreprocessor
        from src.train import ModelTrainer, build_baseline_model

        report: dict = {"success": False, "error": None}

        if not os.path.exists(new_data_path):
            report["error"] = f"Data file not found: {new_data_path}"
            return report

        # Load data
        df = pd.read_csv(new_data_path)
        pre = DataPreprocessor()
        X, y = pre.prepare_features(df)

        if len(y) < 20:
            report["error"] = "Insufficient data for retraining (need ≥ 20 rows)."
            return report

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        evaluator = ModelEvaluator()

        # Evaluate old model
        old_metrics = {}
        if os.path.exists(old_model_path):
            old_model = joblib.load(old_model_path)
            old_metrics = evaluator.evaluate_model(old_model, X_test, y_test)

        # Train new model
        trainer = ModelTrainer()
        all_metrics = {}
        trained = {}
        for name in trainer.get_models():
            try:
                best_pipe, _ = trainer.hyperparameter_search(X_train, y_train, name)
                metrics = evaluator.evaluate_model(best_pipe, X_test, y_test)
                all_metrics[name] = metrics
                trained[name] = best_pipe
            except Exception:
                pass

        if not trained:
            report["error"] = "No models trained successfully."
            return report

        best_name, best_metrics = trainer.select_best_model(all_metrics)
        new_model = trained[best_name]

        # Validate: new model must be at least as good as the old one
        improved = (
            best_metrics.get("f1_macro", 0) >= old_metrics.get("f1_macro", 0)
        )

        if improved:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump(new_model, output_path)

        report.update(
            {
                "success": True,
                "best_new_model": best_name,
                "old_metrics": old_metrics,
                "new_metrics": best_metrics,
                "improved": improved,
                "saved_to": output_path if improved else None,
                "message": (
                    f"New model saved to {output_path}."
                    if improved
                    else "New model did not improve over old model; NOT replaced."
                ),
            }
        )
        return report
