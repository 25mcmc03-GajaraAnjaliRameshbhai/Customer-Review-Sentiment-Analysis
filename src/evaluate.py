"""
src/evaluate.py
---------------
Standalone evaluation helpers for ReviewLens.
"""

from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def full_evaluation(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Return a comprehensive evaluation dictionary.

    Parameters
    ----------
    y_true : list[str]   Ground-truth labels.
    y_pred : list[str]   Predicted labels.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, report (str), confusion_matrix (list)
    """
    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":         round(float(f1_score(y_true, y_pred, average="macro")),    4),
        "weighted_f1":      round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "report":           classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
