"""
train.py
========
Multi-architecture model training, threshold sweep, comparison, and
persistence.

Trains and compares FOUR candidate architectures for chess match outcome
prediction:
- XGBoost-Multiclass : a single tuned multi:softprob XGBoost classifier.
- XGBoost-TwoStage   : stage 1 decides draw (1) vs not-draw (0), stage 2
  decides white (0) vs black (1) among non-draw games; the draw threshold
  t1 is chosen by a validation macro-F1 sweep and the colour threshold t2
  stays at 0.5.
- RandomForest       : a plain multiclass RandomForest.
- Baseline-RatingOnly: a LogisticRegression using only rating columns.

All four are evaluated on the SAME held-out test split and the best model
is selected AUTOMATICALLY (highest accuracy, ties broken by lower log-loss)
and persisted whole to models/best_model.pkl with metadata.  The deployed
artifact always exposes a uniform interface (predict, predict_proba with
columns [win_white, win_black, draw], feature_names) so that prediction.py
and SHAP are architecture-agnostic.

Usage
-----
    python -m src.train
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Allow running as script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import (
    TARGET_DRAW,
    TARGET_WIN_BLACK,
    TARGET_WIN_WHITE,
)
from src.preprocessing import DataPreprocessor

RANDOM_STATE = 42
T1_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]
T2_DEFAULT = 0.5


# --------------------------------------------------------------------------
# Two-stage model wrapper
# --------------------------------------------------------------------------

class TwoStageModel:
    """
    Picklable wrapper combining two binary pipelines into a 3-class model.

    The three target classes are 0=win_white, 1=win_black, 2=draw.

    Attributes
    ----------
    stage1 : fitted Pipeline
        Binary pipeline predicting draw (1) vs not-draw (0).
    stage2 : fitted Pipeline
        Binary pipeline predicting white (0) vs black (1) on non-draw games.
    t1 : float
        Draw decision threshold applied to P(draw).
    t2 : float
        Colour decision threshold applied to P(win_white | not draw).
    feature_names : list of str
        The ordered Option-A feature names.
    classes_ : np.ndarray
        Always np.array([0, 1, 2]).
    """

    def __init__(self, stage1, stage2, t1, t2, feature_names):
        self.stage1 = stage1
        self.stage2 = stage2
        self.t1 = t1
        self.t2 = t2
        self.feature_names = list(feature_names)
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X) -> np.ndarray:
        """
        Return a 3-class probability matrix.

        Columns are ordered [win_white, win_black, draw] to match classes_.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix with the Option-A feature columns.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)
            Rows sum to 1.
        """
        draw_idx = list(self.stage1.classes_).index(1)
        white_idx = list(self.stage2.classes_).index(0)

        p_draw = self.stage1.predict_proba(X)[:, draw_idx]
        p_white = self.stage2.predict_proba(X)[:, white_idx]

        p_win_white = (1.0 - p_draw) * p_white
        p_win_black = (1.0 - p_draw) * (1.0 - p_white)

        return np.column_stack([p_win_white, p_win_black, p_draw])

    def predict(self, X) -> np.ndarray:
        """
        Return 3-class hard predictions using the two thresholds.

        A sample is a draw (2) when P(draw) >= t1; otherwise it is white (0)
        when P(win_white | not draw) >= t2, else black (1).

        Parameters
        ----------
        X : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        draw_idx = list(self.stage1.classes_).index(1)
        white_idx = list(self.stage2.classes_).index(0)

        p_draw = self.stage1.predict_proba(X)[:, draw_idx]
        p_white = self.stage2.predict_proba(X)[:, white_idx]

        preds = np.where(
            p_draw >= self.t1,
            TARGET_DRAW,
            np.where(p_white >= self.t2, TARGET_WIN_WHITE, TARGET_WIN_BLACK),
        )
        return preds.astype(int)


# --------------------------------------------------------------------------
# Single multiclass model adapter
# --------------------------------------------------------------------------

class MulticlassModel:
    """
    Picklable adapter wrapping a single multiclass pipeline.

    Mirrors the TwoStageModel interface so the serving path (prediction.py)
    and SHAP stay architecture-agnostic: both expose predict, predict_proba
    with columns ordered [win_white, win_black, draw], and feature_names.

    The underlying XGBClassifier is trained with objective 'multi:softprob'
    and num_class=3 on the 3-class target (0=win_white, 1=win_black,
    2=draw), so its native class order already matches the required column
    order and no reordering is needed.

    Attributes
    ----------
    pipeline : fitted Pipeline
        Preprocessing + multiclass XGBClassifier.
    feature_names : list of str
        The ordered Option-A feature names.
    classes_ : np.ndarray
        Always np.array([0, 1, 2]).
    """

    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X) -> np.ndarray:
        """
        Return a 3-class probability matrix.

        Columns are ordered [win_white, win_black, draw] to match classes_.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix with the Option-A feature columns.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)
            Rows sum to 1.
        """
        proba = self.pipeline.predict_proba(X)
        # Reorder to [0, 1, 2] in case the classifier's class order differs.
        order = [list(self.pipeline.classes_).index(c) for c in (0, 1, 2)]
        return proba[:, order]

    def predict(self, X) -> np.ndarray:
        """
        Return 3-class hard predictions (argmax of predict_proba).

        Parameters
        ----------
        X : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self.predict_proba(X).argmax(axis=1).astype(int)


# --------------------------------------------------------------------------
# Pipeline builders
# --------------------------------------------------------------------------

def _build_xgb_pipeline(scale_pos_weight=None) -> Pipeline:
    """
    Build a preprocessing + XGBClassifier binary pipeline.

    Parameters
    ----------
    scale_pos_weight : float or None
        Class imbalance weight passed to XGBClassifier (used on stage 1).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = DataPreprocessor().create_preprocessing_pipeline()
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


def _build_random_forest_pipeline() -> Pipeline:
    """Build a plain multiclass RandomForest comparison pipeline."""
    preprocessor = DataPreprocessor().create_preprocessing_pipeline()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


def _build_multiclass_pipeline() -> Pipeline:
    """
    Build the single multiclass XGBoost pipeline (multi:softprob, 3 classes).

    Hyperparameters were tuned via 5-fold stratified CV on the real dataset
    and verified to beat the two-stage model on accuracy and, decisively, on
    log-loss (much better-calibrated probabilities).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = DataPreprocessor().create_preprocessing_pipeline()
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


def build_baseline_model() -> Pipeline:
    """
    Logistic regression using ONLY the rating-related Option-A columns.

    Used to check whether the richer feature set and the two-stage design
    add value over a plain rating-difference baseline.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    baseline_features = ["white_rating", "black_rating", "rating_diff"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, baseline_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

def train_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
) -> Tuple[TwoStageModel, float, Dict[float, float]]:
    """
    Fit both stages and choose t1 by a validation macro-F1 sweep.

    Parameters
    ----------
    X_train, y_train : training fold (3-class labels)
    X_val, y_val : validation fold used only for the t1 sweep
    feature_names : ordered Option-A feature names

    Returns
    -------
    (model, chosen_t1, sweep_scores)
        model : fitted TwoStageModel with the selected t1
        chosen_t1 : float
        sweep_scores : dict of {t1_candidate: validation macro-F1}
    """
    from sklearn.metrics import f1_score

    # Stage 1: draw (1) vs not-draw (0)
    y_stage1 = (y_train == TARGET_DRAW).astype(int)
    n_draw = int((y_stage1 == 1).sum())
    n_not_draw = int((y_stage1 == 0).sum())
    scale_pos_weight = n_not_draw / max(n_draw, 1)

    stage1 = _build_xgb_pipeline(scale_pos_weight=scale_pos_weight)
    stage1.fit(X_train, y_stage1)

    # Stage 2: white (0) vs black (1) on non-draw rows only
    non_draw_mask = y_train != TARGET_DRAW
    X_stage2 = X_train[non_draw_mask]
    y_stage2 = y_train[non_draw_mask].astype(int)  # already 0=white, 1=black
    stage2 = _build_xgb_pipeline()
    stage2.fit(X_stage2, y_stage2)

    # t1 sweep on the validation split
    model = TwoStageModel(stage1, stage2, T1_CANDIDATES[0], T2_DEFAULT, feature_names)
    sweep_scores: Dict[float, float] = {}
    best_t1 = T1_CANDIDATES[0]
    best_f1 = -1.0
    for t1 in T1_CANDIDATES:
        model.t1 = t1
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        sweep_scores[t1] = float(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t1 = t1

    model.t1 = best_t1
    return model, best_t1, sweep_scores


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_model(model, metadata: dict, model_path: str, metadata_path: str):
    """Persist a fitted model and its metadata."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


def _jsonable_metrics(metrics: dict) -> dict:
    """Drop non-serialisable values (e.g. numpy arrays) from a metrics dict."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------

def select_best_model(all_results: Dict[str, dict]) -> str:
    """
    Pick the deployed model name: highest accuracy, ties broken by log-loss.

    Parameters
    ----------
    all_results : dict of {model_name: metrics}
        Each metrics dict must carry 'accuracy' and 'log_loss'.

    Returns
    -------
    str
        The selected model name.
    """
    def _key(name):
        m = all_results[name]
        # Higher accuracy first; among ties, lower log-loss wins.
        return (m.get("accuracy", 0.0), -m.get("log_loss", 9999.0))

    return max(all_results, key=_key)


def main():
    """Run the full four-candidate training, comparison, and persistence."""
    from src.evaluate import ModelEvaluator

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "dataset.csv"
    )
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

    if not os.path.exists(DATA_PATH):
        print("\n" + "=" * 60)
        print("DATASET NOT FOUND")
        print("=" * 60)
        print(f"Expected: {DATA_PATH}")
        print("\nTo build the training dataset, run:")
        print("    python -m src.data_loader")
        print("=" * 60)
        sys.exit(0)

    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {df.shape[1]} columns")

    pre = DataPreprocessor()
    X, y = pre.prepare_features(df)
    feature_names = list(X.columns)
    print(f"  {X.shape[1]} features, {len(y)} samples")
    print(f"  Class distribution: {dict(y.value_counts())}")

    if len(y) < 20:
        print("\nInsufficient data for training (need at least 20 rows).")
        sys.exit(0)

    # Stratified 80/20 train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Carve a stratified validation split out of train for the t1 sweep
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    print(
        f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} rows"
    )

    evaluator = ModelEvaluator()
    all_results: Dict[str, dict] = {}
    trained_models: Dict[str, object] = {}

    # --- Single multiclass XGBoost ---
    print("\n--- XGBoost multiclass (multi:softprob) ---")
    mc_pipeline = _build_multiclass_pipeline()
    mc_pipeline.fit(X_train_full, y_train_full)
    multiclass = MulticlassModel(mc_pipeline, feature_names)
    mc_metrics = evaluator.evaluate_model(multiclass, X_test, y_test)
    all_results["XGBoost-Multiclass"] = mc_metrics
    trained_models["XGBoost-Multiclass"] = multiclass
    print(
        f"  Acc={mc_metrics['accuracy']:.4f}  "
        f"F1={mc_metrics['f1_macro']:.4f}  "
        f"LogLoss={mc_metrics['log_loss']:.4f}  "
        f"RecallEmpate={mc_metrics['recall_draw']:.4f}"
    )

    # --- Two-stage XGBoost (comparison architecture) ---
    print("\n--- Two-stage XGBoost ---")
    two_stage, chosen_t1, sweep_scores = train_two_stage(
        X_train, y_train, X_val, y_val, feature_names
    )
    print(f"  t1 sweep (val macro-F1): {sweep_scores}")
    print(f"  chosen t1 = {chosen_t1}  (t2 = {T2_DEFAULT})")
    two_stage_metrics = evaluator.evaluate_model(two_stage, X_test, y_test)
    all_results["XGBoost-TwoStage"] = two_stage_metrics
    trained_models["XGBoost-TwoStage"] = two_stage
    print(
        f"  Acc={two_stage_metrics['accuracy']:.4f}  "
        f"F1={two_stage_metrics['f1_macro']:.4f}  "
        f"LogLoss={two_stage_metrics['log_loss']:.4f}  "
        f"RecallEmpate={two_stage_metrics['recall_draw']:.4f}"
    )

    # --- RandomForest (plain multiclass) ---
    print("\n--- RandomForest (multiclass) ---")
    rf = _build_random_forest_pipeline()
    rf.fit(X_train_full, y_train_full)
    rf_metrics = evaluator.evaluate_model(rf, X_test, y_test)
    all_results["RandomForest"] = rf_metrics
    trained_models["RandomForest"] = rf
    print(
        f"  Acc={rf_metrics['accuracy']:.4f}  "
        f"F1={rf_metrics['f1_macro']:.4f}  "
        f"LogLoss={rf_metrics['log_loss']:.4f}"
    )

    # --- Baseline (rating only) ---
    print("\n--- Baseline (LogisticRegression, rating only) ---")
    baseline = build_baseline_model()
    baseline.fit(X_train_full, y_train_full)
    baseline_metrics = evaluator.evaluate_model(baseline, X_test, y_test)
    all_results["Baseline-RatingOnly"] = baseline_metrics
    trained_models["Baseline-RatingOnly"] = baseline
    print(
        f"  Acc={baseline_metrics['accuracy']:.4f}  "
        f"F1={baseline_metrics['f1_macro']:.4f}  "
        f"LogLoss={baseline_metrics['log_loss']:.4f}"
    )

    # Comparison table
    comparison_df = evaluator.compare_models(all_results)
    print("\n" + comparison_df.to_string())

    # --- Automatic model selection: highest accuracy, ties -> lower log-loss
    best_name = select_best_model(all_results)
    best_model = trained_models[best_name]
    best_metrics = all_results[best_name]
    print(
        f"\nSelected deployed model: {best_name} "
        f"(Acc={best_metrics['accuracy']:.4f}, "
        f"LogLoss={best_metrics['log_loss']:.4f})"
    )

    # --- Confusion matrices + draw metrics for the DEPLOYED model
    y_pred_best = best_model.predict(X_test)
    conf_matrix = evaluator.confusion_matrix_3x3(y_test, y_pred_best)
    draw_metrics = evaluator.draw_confusion(y_test, y_pred_best)

    metadata = {
        "algorithm": best_name,
        "training_date": datetime.now().isoformat(),
        "train_samples": len(X_train_full),
        "test_samples": len(X_test),
        "metrics": _jsonable_metrics(best_metrics),
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_labels": ["win_white", "win_black", "draw"],
        "draw_metrics": draw_metrics,
        "feature_names": feature_names,
        "all_model_results": {
            name: _jsonable_metrics(metrics)
            for name, metrics in all_results.items()
        },
    }
    # The two-stage architecture keeps its thresholds / sweep in the
    # comparison table so its conceptual value is preserved in metadata.
    metadata["all_model_results"]["XGBoost-TwoStage"]["thresholds"] = {
        "t1": chosen_t1,
        "t2": T2_DEFAULT,
    }
    metadata["all_model_results"]["XGBoost-TwoStage"]["chosen_t1"] = chosen_t1
    metadata["all_model_results"]["XGBoost-TwoStage"]["t1_sweep"] = {
        str(k): v for k, v in sweep_scores.items()
    }
    # When the two-stage model is the deployed one, surface its thresholds
    # at the top level too (kept for backwards compatibility with the app).
    if best_name == "XGBoost-TwoStage":
        metadata["thresholds"] = {"t1": chosen_t1, "t2": T2_DEFAULT}
        metadata["chosen_t1"] = chosen_t1
        metadata["t1_sweep"] = {str(k): v for k, v in sweep_scores.items()}

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    save_model(best_model, metadata, model_path, meta_path)

    # --- Plots
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    comparison_fig_path = os.path.join(plots_dir, "model_comparison.png")
    comp_fig = evaluator.plot_model_comparison(
        comparison_df, save_path=comparison_fig_path
    )
    plt.close(comp_fig)
    print(f"\nComparison plot saved to {comparison_fig_path}")

    cm_fig_path = os.path.join(plots_dir, "confusion_matrix.png")
    cm_fig = evaluator.plot_confusion_matrix_counts(
        conf_matrix, best_name, save_path=cm_fig_path
    )
    plt.close(cm_fig)
    print(f"Confusion matrix saved to {cm_fig_path}")

    cm_draw_fig_path = os.path.join(plots_dir, "confusion_matrix_draw.png")
    cm_draw_fig = evaluator.plot_draw_confusion(
        draw_metrics, best_name, save_path=cm_draw_fig_path
    )
    plt.close(cm_draw_fig)
    print(f"Draw confusion matrix saved to {cm_draw_fig_path}")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Delegate to the package module so persisted TwoStageModel instances
    # carry __module__ == 'src.train' and reload cleanly anywhere via joblib,
    # rather than being bound to '__main__'.
    from src.train import main as _main

    _main()
