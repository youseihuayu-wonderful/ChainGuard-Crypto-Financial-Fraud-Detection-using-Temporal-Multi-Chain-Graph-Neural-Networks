"""
Evaluation metrics for fraud detection.

All metrics work with binary classification:
    - Class 1: illicit (positive class — what we want to detect)
    - Class 0: licit (negative class)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all fraud detection metrics.

    Args:
        y_true: Ground truth labels (1=illicit, 0=licit).
        y_prob: Predicted probability of being illicit.
        threshold: Classification threshold.

    Returns:
        Dictionary of metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    return metrics


def print_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Print a full classification report."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_prob, threshold)
    print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"F1 (illicit): {metrics['f1']:.4f}")
    print(f"Precision (illicit): {metrics['precision']:.4f}")
    print(f"Recall (illicit): {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              pred_licit  pred_illicit")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  true_licit    {cm[0][0]:>6d}      {cm[0][1]:>6d}")
    print(f"  true_illicit  {cm[1][0]:>6d}      {cm[1][1]:>6d}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['licit', 'illicit'])}")
