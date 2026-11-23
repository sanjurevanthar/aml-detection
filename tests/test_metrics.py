"""Checks for the shared evaluation metrics."""

import unittest

import numpy as np

from src.metrics import (
    confusion_at_threshold,
    ranking_metrics,
    threshold_from_val,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        scores = y.astype(float)
        m = ranking_metrics(y, scores, top_fraction=0.5, top_k=3)
        self.assertEqual(m["recall_top_1pct"], 1.0)
        self.assertEqual(m["precision_top_1pct"], 1.0)
        self.assertEqual(m["precision_top_100"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)

    def test_threshold_and_confusion(self):
        y_val = np.array([0, 0, 1, 0])
        scores_val = np.array([0.1, 0.2, 0.9, 0.3])
        cutoff = threshold_from_val(y_val, scores_val, top_fraction=0.5)
        self.assertAlmostEqual(cutoff, 0.3)
        cm = confusion_at_threshold(y_val, scores_val, cutoff)
        self.assertEqual(cm["tp"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["fn"], 0)
        self.assertAlmostEqual(cm["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
