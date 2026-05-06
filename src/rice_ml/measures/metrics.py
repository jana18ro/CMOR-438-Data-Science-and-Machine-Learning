"""
metrics.py

Evaluation metrics for the rice_ml package.

This module contains simple NumPy implementations of common classification and
regression metrics. The functions are intentionally lightweight so they can be
used in tests, notebooks, and from-scratch model implementations without
depending on scikit-learn.

Classification metrics
----------------------
- accuracy_score
- precision
- recall
- f1_score
- confusion_matrix

Regression metrics
------------------
- mse
- rmse
- mae
- r2_score
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]


def _as_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input into a non-empty 1D NumPy array."""
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input into a non-empty 1D float NumPy array."""
    return _as_1d_array(x, name).astype(float)


def _check_same_length(y_true: ArrayLike, y_pred: ArrayLike):
    """Validate that y_true and y_pred are aligned 1D arrays."""
    yt = _as_1d_array(y_true, "y_true")
    yp = _as_1d_array(y_pred, "y_pred")

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")

    return yt, yp


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of correctly predicted labels."""
    yt, yp = _check_same_length(y_true, y_pred)
    return float(np.mean(yt == yp))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[Sequence] = None) -> np.ndarray:
    """
    Build a confusion matrix.

    Rows represent true classes. Columns represent predicted classes.
    """
    yt, yp = _check_same_length(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))

    labels = np.asarray(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, pred_label in zip(yt, yp):
        if true_label in label_to_index and pred_label in label_to_index:
            matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix


def _precision_recall_f1_by_class(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence] = None):
    """Compute per-class precision, recall, and F1 arrays."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_values = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall_values = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1_values = np.where(
            precision_values + recall_values > 0,
            2 * precision_values * recall_values / (precision_values + recall_values),
            0.0,
        )

    return precision_values, recall_values, f1_values


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label=None,
    labels: Optional[Sequence] = None,
):
    """
    Compute precision.

    average options:
    - "binary": return precision for the positive class
    - "macro": return the unweighted mean across classes
    - None: return per-class precision values
    """
    yt, yp = _check_same_length(y_true, y_pred)

    if average == "binary":
        unique_labels = np.unique(np.concatenate([yt, yp]))

        if positive_label is None:
            if len(unique_labels) != 2:
                raise ValueError("positive_label is required when the task is not binary.")
            positive_label = unique_labels[-1]

        tp = np.sum((yt == positive_label) & (yp == positive_label))
        fp = np.sum((yt != positive_label) & (yp == positive_label))

        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    precision_values, _, _ = _precision_recall_f1_by_class(yt, yp, labels=labels)

    if average == "macro":
        return float(np.mean(precision_values))

    if average is None:
        return precision_values

    raise ValueError("average must be 'binary', 'macro', or None.")


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label=None,
    labels: Optional[Sequence] = None,
):
    """
    Compute recall.

    average options:
    - "binary": return recall for the positive class
    - "macro": return the unweighted mean across classes
    - None: return per-class recall values
    """
    yt, yp = _check_same_length(y_true, y_pred)

    if average == "binary":
        unique_labels = np.unique(np.concatenate([yt, yp]))

        if positive_label is None:
            if len(unique_labels) != 2:
                raise ValueError("positive_label is required when the task is not binary.")
            positive_label = unique_labels[-1]

        tp = np.sum((yt == positive_label) & (yp == positive_label))
        fn = np.sum((yt == positive_label) & (yp != positive_label))

        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    _, recall_values, _ = _precision_recall_f1_by_class(yt, yp, labels=labels)

    if average == "macro":
        return float(np.mean(recall_values))

    if average is None:
        return recall_values

    raise ValueError("average must be 'binary', 'macro', or None.")


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label=None,
    labels: Optional[Sequence] = None,
):
    """
    Compute the F1 score, the harmonic mean of precision and recall.
    """
    yt, yp = _check_same_length(y_true, y_pred)

    if average == "binary":
        p = precision(yt, yp, average="binary", positive_label=positive_label)
        r = recall(yt, yp, average="binary", positive_label=positive_label)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    _, _, f1_values = _precision_recall_f1_by_class(yt, yp, labels=labels)

    if average == "macro":
        return float(np.mean(f1_values))

    if average is None:
        return f1_values

    raise ValueError("average must be 'binary', 'macro', or None.")


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared error."""
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")

    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error."""
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")

    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the R-squared coefficient of determination."""
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)


# Common aliases used in some notebooks.
mean_squared_error = mse
root_mean_squared_error = rmse
accuracy = accuracy_score
