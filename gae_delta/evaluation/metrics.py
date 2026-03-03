"""Evaluation metrics: AUC-ROC, F1, precision, recall."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation results."""
    auc_roc: float
    f1: float
    precision: float
    recall: float
    threshold: float


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> ClassificationMetrics:
    """Compute classification metrics at the optimal operating point.

    The optimal threshold is chosen to maximize Youden's J statistic
    (sensitivity + specificity - 1) on the ROC curve.

    Parameters
    ----------
    y_true : (n,) binary labels
    y_prob : (n,) predicted probabilities of positive class (poor outcome)

    Returns
    -------
    ClassificationMetrics
    """
    auc = roc_auc_score(y_true, y_prob)

    # Find optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])

    # Binarize predictions at optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)

    return ClassificationMetrics(
        auc_roc=float(auc),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=optimal_threshold,
    )
