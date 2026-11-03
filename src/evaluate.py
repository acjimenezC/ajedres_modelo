"""
evaluate.py
===========
Model evaluation utilities.

Computes accuracy, precision, recall, F1 (macro), log-loss, and
confusion matrices.  Provides model comparison tables and probability
calibration.
"""

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

TARGET_LABELS = ["Victoria Blancas", "Victoria Negras", "Empate"]

# Spanish tick labels for the persisted confusion-matrix images
CM_TICK_LABELS = ["Gana Blancas", "Gana Negras", "Tablas"]
DRAW_TICK_LABELS = ["No Tablas", "Tablas"]
DRAW_CLASS = 2


class ModelEvaluator:
    """
    Evaluates trained sklearn pipelines and generates comparison artifacts.
    """

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, model, X_test, y_test) -> dict:
        """
        Compute all evaluation metrics for a fitted model.

        Parameters
        ----------
        model : sklearn estimator with predict / predict_proba
        X_test, y_test : test split

        Returns
        -------
        dict with keys: accuracy, precision_macro, recall_macro, f1_macro,
                        log_loss, recall_draw, classification_report,
                        confusion_matrix
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        ll = log_loss(y_test, y_proba, labels=[0, 1, 2])
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["win_white", "win_black", "draw"],
            zero_division=0,
        )

        # Recall for draw class (index 2)
        draw_classes = np.unique(y_test)
        if 2 in draw_classes:
            recall_draw = recall_score(
                y_test, y_pred, labels=[2], average="macro", zero_division=0
            )
        else:
            recall_draw = 0.0

        # Verify probabilities sum to 1
        prob_sums = y_proba.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1!"

        return {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "log_loss": float(ll),
            "recall_draw": float(recall_draw),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_models(self, results: Dict[str, dict]) -> pd.DataFrame:
        """
        Build a comparison DataFrame from a dict of {name: metrics}.

        Returns
        -------
        pd.DataFrame sorted by f1_macro descending.
        """
        rows = []
        for name, metrics in results.items():
            rows.append(
                {
                    "Modelo": name,
                    "Accuracy": round(metrics.get("accuracy", 0), 4),
                    "Precision Macro": round(metrics.get("precision_macro", 0), 4),
                    "Recall Macro": round(metrics.get("recall_macro", 0), 4),
                    "F1 Macro": round(metrics.get("f1_macro", 0), 4),
                    "Log-loss": round(metrics.get("log_loss", 9999), 4),
                    "Recall Empate": round(metrics.get("recall_draw", 0), 4),
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values("F1 Macro", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Confusion matrices ("tabla de verdad")
    # ------------------------------------------------------------------

    def confusion_matrix_3x3(self, y_true, y_pred) -> np.ndarray:
        """
        Full 3x3 confusion matrix (rows=true, cols=pred).

        Label order is [win_white=0, win_black=1, draw=2].

        Parameters
        ----------
        y_true, y_pred : array-like of int class labels

        Returns
        -------
        np.ndarray of shape (3, 3) with integer counts.
        """
        return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    def draw_confusion(self, y_true, y_pred) -> dict:
        """
        Binary "draw vs not-draw" confusion matrix and derived metrics.

        The positive class is "draw" (label 2); everything else is the
        negative class.

        Parameters
        ----------
        y_true, y_pred : array-like of int class labels (0/1/2)

        Returns
        -------
        dict
            {tp, fn, fp, tn, precision, recall, f1}, where
            tp = true draws predicted draw,
            fn = true draws predicted not-draw,
            fp = true not-draws predicted draw,
            tn = true not-draws predicted not-draw.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        true_draw = y_true == DRAW_CLASS
        pred_draw = y_pred == DRAW_CLASS

        tp = int(np.sum(true_draw & pred_draw))
        fn = int(np.sum(true_draw & ~pred_draw))
        fp = int(np.sum(~true_draw & pred_draw))
        tn = int(np.sum(~true_draw & ~pred_draw))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_confusion_matrix_counts(
        self,
        conf_matrix,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render a 3x3 confusion matrix of raw counts as a heatmap.

        Spanish labels: eje y = "Real", eje x = "Predicho", class ticks
        "Gana Blancas" / "Gana Negras" / "Tablas".

        Parameters
        ----------
        conf_matrix : array-like of shape (3, 3)
            Count matrix (rows=true, cols=pred), label order [0, 1, 2].
        model_name : str
            Deployed model name, used in the title.
        save_path : str or None
            If given, the figure is written to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        cm = np.asarray(conf_matrix)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CM_TICK_LABELS,
            yticklabels=CM_TICK_LABELS,
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=13)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
        return fig

    def plot_draw_confusion(
        self,
        draw_metrics: dict,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render the 2x2 "draw vs not-draw" confusion matrix as a heatmap.

        Spanish labels: eje y = "Real", eje x = "Predicho", ticks
        "No Tablas" / "Tablas".  Rows/cols order is [No Tablas, Tablas].

        Parameters
        ----------
        draw_metrics : dict
            Output of ``draw_confusion`` (tp/fn/fp/tn + metrics).
        model_name : str
            Deployed model name, used in the title.
        save_path : str or None
            If given, the figure is written to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        tp = draw_metrics["tp"]
        fn = draw_metrics["fn"]
        fp = draw_metrics["fp"]
        tn = draw_metrics["tn"]
        # rows = Real [No Tablas, Tablas], cols = Predicho [No Tablas, Tablas]
        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=DRAW_TICK_LABELS,
            yticklabels=DRAW_TICK_LABELS,
            ax=ax,
        )
        ax.set_title(
            f"Detección de Tablas — {model_name}\n"
            f"(P={draw_metrics['precision']:.3f}  "
            f"R={draw_metrics['recall']:.3f}  "
            f"F1={draw_metrics['f1']:.3f})",
            fontsize=12,
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
        return fig

    def plot_confusion_matrix(
        self,
        model,
        X_test,
        y_test,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Normalised confusion matrix heatmap."""
        cm = confusion_matrix(
            y_test, model.predict(X_test), labels=[0, 1, 2], normalize="true"
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=TARGET_LABELS,
            yticklabels=TARGET_LABELS,
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=13)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart comparing F1 macro and Log-loss across models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Comparación de Modelos", fontsize=14, fontweight="bold")

        # F1 macro
        sns.barplot(
            data=comparison_df,
            x="F1 Macro",
            y="Modelo",
            ax=axes[0],
            palette="Blues_d",
        )
        axes[0].set_title("F1 Macro (mayor = mejor)")
        axes[0].set_xlim(0, 1)
        for bar in axes[0].patches:
            axes[0].text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                fontsize=9,
            )

        # Log-loss
        sns.barplot(
            data=comparison_df,
            x="Log-loss",
            y="Modelo",
            ax=axes[1],
            palette="Reds_d",
        )
        axes[1].set_title("Log-loss (menor = mejor)")
        for bar in axes[1].patches:
            axes[1].text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
        return fig

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_probabilities(self, model, X_val, y_val):
        """
        Calibrate a fitted model with CalibratedClassifierCV (cv='prefit').

        Verifies that output probabilities sum to 1.0 ± 1e-6.

        Returns
        -------
        CalibratedClassifierCV fitted on (X_val, y_val)
        """
        calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        calibrated.fit(X_val, y_val)

        # Sanity check
        proba = calibrated.predict_proba(X_val)
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), "Calibrated probs do not sum to 1!"

        return calibrated

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_evaluation_report(self, results: dict):
        """Pretty-print evaluation results to stdout."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        for name, metrics in results.items():
            print(f"\n{name}")
            print("-" * 40)
            print(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision Macro:{metrics.get('precision_macro', 0):.4f}")
            print(f"  Recall Macro:   {metrics.get('recall_macro', 0):.4f}")
            print(f"  F1 Macro:       {metrics.get('f1_macro', 0):.4f}")
            print(f"  Log-loss:       {metrics.get('log_loss', 9999):.4f}")
            print(f"  Recall (Empate):{metrics.get('recall_draw', 0):.4f}")
            if "classification_report" in metrics:
                print("\n  Classification Report:")
                for line in metrics["classification_report"].splitlines():
                    print(f"    {line}")
