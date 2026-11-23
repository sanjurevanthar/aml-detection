"""Metrics shared by the rule baseline and the ML models.

The primary question is about a ranked list, so recall inside the top 1% is the
headline number. PR-AUC and ROC-AUC describe overall ranking quality; accuracy
is deliberately not used because ~0.1% of transactions are laundering.
"""

from __future__ import annotations

from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_n(n_rows: int, top_fraction: float) -> int:
    return max(1, int(ceil(n_rows * top_fraction)))


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    top_fraction: float,
    top_k: int,
) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    ranking = np.argsort(-s, kind="mergesort")
    n_top = _top_n(len(y), top_fraction)
    top = ranking[:n_top]
    n_positive = int(y.sum())

    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < n_positive < len(y) else float("nan"),
        "recall_top_1pct": float(y[top].sum() / n_positive) if n_positive else 0.0,
        "precision_top_1pct": float(y[top].mean()),
        "precision_top_100": float(y[ranking[: min(top_k, len(y))]].mean()),
        "flagged_top_n": int(n_top),
        "positives": n_positive,
        "rows": int(len(y)),
    }


def threshold_from_val(y_val: np.ndarray, scores_val: np.ndarray, top_fraction: float) -> float:
    """Pick the score cutoff that flags the riskiest ``top_fraction`` of val rows."""
    s = np.asarray(scores_val, dtype=np.float64)
    ranking = np.argsort(-s, kind="mergesort")
    return float(s[ranking[_top_n(len(s), top_fraction) - 1]])


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Confusion matrix at a fixed score cutoff, plus precision/recall/F1."""
    y = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
