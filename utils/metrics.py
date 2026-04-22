"""
Evaluation metrics for recommender systems.

Implements:
  - AUC: Area Under the ROC Curve
  - UAUC: User-level AUC (averaged across users)
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC).

    Args:
        predictions: Predicted probabilities, shape (N,).
        labels: Binary ground truth labels, shape (N,).

    Returns:
        AUC score. Returns 0.5 if only one class is present.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, predictions)


def compute_uauc(
    predictions: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
) -> float:
    """Compute User-level AUC (UAUC).

    Computes AUC independently for each user and returns the weighted
    average (weighted by number of samples per user).

    Args:
        predictions: Predicted probabilities, shape (N,).
        labels: Binary ground truth labels, shape (N,).
        user_ids: User identifiers, shape (N,).

    Returns:
        Weighted average of per-user AUC scores.
    """
    unique_users = np.unique(user_ids)
    total_weight = 0
    weighted_auc = 0.0

    for uid in unique_users:
        mask = user_ids == uid
        user_labels = labels[mask]
        user_preds = predictions[mask]

        if len(np.unique(user_labels)) < 2:
            continue

        user_auc = roc_auc_score(user_labels, user_preds)
        weight = mask.sum()
        weighted_auc += user_auc * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return weighted_auc / total_weight
