"""
tests/test_evaluate.py
======================
Tests for ModelEvaluator confusion-matrix / draw-detection helpers.

All tests use tiny hand-made label arrays and run fully offline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import ModelEvaluator


class TestDrawConfusion:
    """Tests for the 2x2 draw-vs-not-draw confusion helper."""

    def setup_method(self):
        self.ev = ModelEvaluator()

    def test_draw_confusion_counts_and_metrics(self):
        """Hand-built labels: verify TP/FN/FP/TN and precision/recall/f1.

        Labels (0=win_white, 1=win_black, 2=draw):
            true: 2 2 2 0 1 2 0
            pred: 2 0 1 2 1 2 0
        Draw is the positive class (label 2).
          - true draws (y_true==2): indices 0,1,2,5  -> 4 total
              predicted draw at 0 and 5  -> TP = 2
              predicted not-draw at 1,2  -> FN = 2
          - true not-draws: indices 3,4,6 -> 3 total
              predicted draw at 3        -> FP = 1
              predicted not-draw at 4,6  -> TN = 2
        """
        y_true = [2, 2, 2, 0, 1, 2, 0]
        y_pred = [2, 0, 1, 2, 1, 2, 0]

        dm = self.ev.draw_confusion(y_true, y_pred)

        assert dm["tp"] == 2
        assert dm["fn"] == 2
        assert dm["fp"] == 1
        assert dm["tn"] == 2

        # precision = TP/(TP+FP) = 2/3, recall = TP/(TP+FN) = 2/4 = 0.5
        assert dm["precision"] == 2 / 3
        assert dm["recall"] == 0.5
        expected_f1 = 2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5)
        assert abs(dm["f1"] - expected_f1) < 1e-9

    def test_draw_confusion_no_predicted_draws(self):
        """When no draw is predicted, precision/recall/f1 degrade to 0."""
        y_true = [2, 2, 0, 1]
        y_pred = [0, 1, 0, 1]
        dm = self.ev.draw_confusion(y_true, y_pred)
        assert dm["tp"] == 0
        assert dm["fn"] == 2
        assert dm["fp"] == 0
        assert dm["tn"] == 2
        assert dm["precision"] == 0.0
        assert dm["recall"] == 0.0
        assert dm["f1"] == 0.0

    def test_confusion_matrix_3x3_shape_and_counts(self):
        """The 3x3 matrix must have rows=true, cols=pred in order [0,1,2]."""
        y_true = [0, 0, 1, 2]
        y_pred = [0, 1, 1, 2]
        cm = self.ev.confusion_matrix_3x3(y_true, y_pred)
        assert cm.shape == (3, 3)
        # row 0 (true win_white): one pred 0, one pred 1
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        # row 1 (true win_black): one pred 1
        assert cm[1, 1] == 1
        # row 2 (true draw): one pred 2
        assert cm[2, 2] == 1
        assert cm.sum() == 4
