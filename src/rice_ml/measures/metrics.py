"""
metrics.py

Evaluation metrics for the rice_ml package.

This module contains simple NumPy implementations of common classification and
regression metrics. All functions are intentionally lightweight so they can be
used in tests, notebooks, and from-scratch model implementations without
depending on scikit-learn.

Classification metrics
----------------------
- accuracy_score          (alias: accuracy)
- precision_score         (alias: precision)
- recall_score            (alias: recall)
- f1_score
- confusion_matrix
- classification_report

Regression metrics
------------------
- mean_squared_error      (alias: mse)
- root_mean_squared_error (alias: rmse)
- mean_absolute_error     (alias: mae)
- r2_score
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, Sequence]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_1d(x: ArrayLike, name: str) -> np.ndarray:
    """Return a non-empty 1-D NumPy array; raises on wrong shape or empty."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _to_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    """Return a non-empty 1-D float NumPy array."""
    return _to_1d(x, name).astype(float)


def _check_lengths(y_true: ArrayLike, y_pred: ArrayLike):
    """Validate and return aligned 1-D arrays of the same length."""
    yt = _to_1d(y_true, "y_true")
    yp = _to_1d(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(yt)} and {len(yp)})."
        )
    return yt, yp


def _resolve_labels(yt: np.ndarray, yp: np.ndarray, labels: Optional[ArrayLike]) -> np.ndarray:
    """Return a sorted array of class labels to use."""
    if labels is None:
        return np.unique(np.concatenate([yt, yp]))
    return np.asarray(labels)


def _validate_average(average: Optional[str]) -> None:
    valid = {"binary", "macro", "micro", "weighted", None}
    if average not in valid:
        raise ValueError(
            f"average must be one of {valid}, got {average!r}."
        )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Build a confusion matrix.

    Rows represent true classes; columns represent predicted classes.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    labels : sequence, optional
        Explicit ordering of classes. Inferred from data when omitted.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
    """
    yt, yp = _check_lengths(y_true, y_pred)
    # Use a local np.ndarray variable so the type is unambiguous to Pylance.
    label_arr: np.ndarray = _resolve_labels(yt, yp, labels)
    n = len(label_arr)

    label_to_idx = {lbl: i for i, lbl in enumerate(label_arr)}
    matrix = np.zeros((n, n), dtype=int)

    for t, p in zip(yt, yp):
        ti = label_to_idx.get(t)  # type: ignore[arg-type]
        pi = label_to_idx.get(p)  # type: ignore[arg-type]
        if ti is not None and pi is not None:
            matrix[ti, pi] += 1

    return matrix


# ---------------------------------------------------------------------------
# Per-class TP / FP / FN counts (shared by precision, recall, f1)
# ---------------------------------------------------------------------------


def _tp_fp_fn(
    yt: np.ndarray,
    yp: np.ndarray,
    label_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-class TP, FP, FN, and support arrays."""
    cm = confusion_matrix(yt, yp, labels=label_arr)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    support = cm.sum(axis=1).astype(float)
    return tp, fp, fn, support


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of correctly predicted labels."""
    yt, yp = _check_lengths(y_true, y_pred)
    return float(np.mean(yt == yp))


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Optional[str] = "binary",
    positive_label=1,
    labels: Optional[ArrayLike] = None,
) -> Union[float, np.ndarray]:
    """
    Compute precision  TP / (TP + FP).

    Parameters
    ----------
    average : {'binary', 'macro', 'micro', 'weighted', None}
        - 'binary'   : precision for ``positive_label`` only (default).
        - 'macro'    : unweighted mean over all classes.
        - 'micro'    : global TP / (TP + FP), treating all classes equally.
        - 'weighted' : mean weighted by class support.
        - None       : return per-class array.
    """
    _validate_average(average)
    yt, yp = _check_lengths(y_true, y_pred)
    label_arr: np.ndarray = _resolve_labels(yt, yp, labels)

    if average == "binary":
        tp = float(np.sum((yt == positive_label) & (yp == positive_label)))
        fp = float(np.sum((yt != positive_label) & (yp == positive_label)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    tp, fp, fn, support = _tp_fp_fn(yt, yp, label_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(tp + fp > 0, tp / (tp + fp), 0.0)

    if average is None:
        return per_class
    if average == "macro":
        return float(per_class.mean())
    if average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        return float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    if average == "weighted":
        return float(np.average(per_class, weights=support)) if support.sum() > 0 else 0.0

    raise ValueError(f"Unhandled average={average!r}")  # unreachable


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Optional[str] = "binary",
    positive_label=1,
    labels: Optional[ArrayLike] = None,
) -> Union[float, np.ndarray]:
    """
    Compute recall  TP / (TP + FN).

    Parameters
    ----------
    average : {'binary', 'macro', 'micro', 'weighted', None}
        Same semantics as ``precision_score``.
    """
    _validate_average(average)
    yt, yp = _check_lengths(y_true, y_pred)
    label_arr: np.ndarray = _resolve_labels(yt, yp, labels)

    if average == "binary":
        tp = float(np.sum((yt == positive_label) & (yp == positive_label)))
        fn = float(np.sum((yt == positive_label) & (yp != positive_label)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    tp, fp, fn, support = _tp_fp_fn(yt, yp, label_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    if average is None:
        return per_class
    if average == "macro":
        return float(per_class.mean())
    if average == "micro":
        tp_sum = tp.sum()
        fn_sum = fn.sum()
        return float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    if average == "weighted":
        return float(np.average(per_class, weights=support)) if support.sum() > 0 else 0.0

    raise ValueError(f"Unhandled average={average!r}")  # unreachable


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Optional[str] = "binary",
    positive_label=1,
    labels: Optional[ArrayLike] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the F1 score — harmonic mean of precision and recall.

    Parameters
    ----------
    average : {'binary', 'macro', 'micro', 'weighted', None}
        Same semantics as ``precision_score``.
    """
    _validate_average(average)
    yt, yp = _check_lengths(y_true, y_pred)
    label_arr: np.ndarray = _resolve_labels(yt, yp, labels)

    if average == "binary":
        p = precision_score(yt, yp, average="binary", positive_label=positive_label)
        r = recall_score(yt, yp, average="binary", positive_label=positive_label)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    tp, fp, fn, support = _tp_fp_fn(yt, yp, label_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        per_class = np.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec),
            0.0,
        )

    if average is None:
        return per_class
    if average == "macro":
        return float(per_class.mean())
    if average == "micro":
        tp_sum  = tp.sum()
        fp_sum  = fp.sum()
        fn_sum  = fn.sum()
        p_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        r_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        return float(2 * p_micro * r_micro / (p_micro + r_micro)) if (p_micro + r_micro) > 0 else 0.0
    if average == "weighted":
        return float(np.average(per_class, weights=support)) if support.sum() > 0 else 0.0

    raise ValueError(f"Unhandled average={average!r}")  # unreachable


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[ArrayLike] = None,
    target_names: Optional[Sequence[str]] = None,
    digits: int = 4,
) -> str:
    """
    Build a text summary of per-class precision, recall, F1, and support.

    Parameters
    ----------
    labels : array-like, optional
        Class labels to include (inferred from data when omitted).
    target_names : list of str, optional
        Human-readable names for each label.
    digits : int
        Number of decimal places for metric values.

    Returns
    -------
    str
        A formatted report string, similar to scikit-learn's output.
    """
    yt, yp = _check_lengths(y_true, y_pred)
    label_arr: np.ndarray = _resolve_labels(yt, yp, labels)
    n_classes = len(label_arr)

    if target_names is None:
        names: List[str] = [str(lbl) for lbl in label_arr]
    else:
        names = list(target_names)
        if len(names) != n_classes:
            raise ValueError(
                f"target_names has {len(names)} entries but there are "
                f"{n_classes} classes."
            )

    tp, fp, fn, support = _tp_fp_fn(yt, yp, label_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_arr = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec_arr  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1_arr   = np.where(
            prec_arr + rec_arr > 0,
            2 * prec_arr * rec_arr / (prec_arr + rec_arr),
            0.0,
        )

    col_w = max(max(len(n) for n in names), 12)
    fmt = f"{{:>{col_w}}}  {{:>{digits + 4}.{digits}f}}  {{:>{digits + 4}.{digits}f}}  {{:>{digits + 4}.{digits}f}}  {{:>8}}"
    header = f"{'':>{col_w}}  {'precision':>{digits + 4}}  {'recall':>{digits + 4}}  {'f1-score':>{digits + 4}}  {'support':>8}"

    lines = [header, ""]
    for i, name in enumerate(names):
        lines.append(fmt.format(name, prec_arr[i], rec_arr[i], f1_arr[i], int(support[i])))

    total_support = int(support.sum())
    lines.append("")

    # Macro averages
    lines.append(fmt.format(
        "macro avg",
        float(prec_arr.mean()),
        float(rec_arr.mean()),
        float(f1_arr.mean()),
        total_support,
    ))

    # Weighted averages
    w = support / support.sum() if support.sum() > 0 else np.ones(n_classes) / n_classes
    lines.append(fmt.format(
        "weighted avg",
        float(np.dot(prec_arr, w)),
        float(np.dot(rec_arr,  w)),
        float(np.dot(f1_arr,   w)),
        total_support,
    ))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared error: mean((y_true - y_pred)²)."""
    yt = _to_1d_float(y_true, "y_true")
    yp = _to_1d_float(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared error: sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error: mean(|y_true - y_pred|)."""
    yt = _to_1d_float(y_true, "y_true")
    yp = _to_1d_float(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Return the coefficient of determination R².

    R² = 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)² and
    SS_tot = Σ(y - ȳ)².  Returns 1.0 when predictions are perfect and
    can be negative when the model is worse than a constant baseline.
    """
    yt = _to_1d_float(y_true, "y_true")
    yp = _to_1d_float(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Short aliases — kept for backward compatibility with notebooks / old code
# ---------------------------------------------------------------------------

accuracy          = accuracy_score
precision         = precision_score
recall            = recall_score
mse               = mean_squared_error
rmse              = root_mean_squared_error
mae               = mean_absolute_error
