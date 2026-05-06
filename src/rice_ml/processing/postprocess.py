"""
postprocess.py

Postprocessing utilities for rice_ml model outputs.

These helpers are meant to be used after a model has produced predictions,
scores, probabilities, or encoded labels. The functions are intentionally small
and dependency-free so they can be used in notebooks, tests, and model demos.

Included tools
--------------
- apply_threshold:
    Convert binary probabilities into class labels.

- one_hot_to_labels:
    Convert one-hot or probability matrices into label indices.

- decode_labels:
    Convert encoded labels back into their original label names.

- top_k_predictions:
    Return the top-k predicted classes from a probability matrix.

- format_predictions:
    Create a simple list of dictionaries containing predictions and optional
    confidence values.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence, Sequence[Sequence]]


def _as_1d_array(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Convert input into a non-empty 1D NumPy array."""
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


def _as_2d_float(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Convert input into a non-empty 2D float NumPy array."""
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")

    return arr


def apply_threshold(probabilities: ArrayLike, threshold: float = 0.5, positive_label=1, negative_label=0) -> np.ndarray:
    """
    Convert binary probabilities into class labels.

    Parameters
    ----------
    probabilities : array-like of shape (n_samples,)
        Probability or score for the positive class.

    threshold : float
        Values greater than or equal to this threshold receive positive_label.

    positive_label, negative_label
        Labels used for the two output classes.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    probs = np.asarray(probabilities, dtype=float)

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D array.")

    return np.where(probs >= threshold, positive_label, negative_label)


def one_hot_to_labels(values: ArrayLike) -> np.ndarray:
    """
    Convert one-hot rows or probability rows into label indices.

    The returned label is the column index with the largest value in each row.
    """
    arr = _as_2d_float(values, name="values")
    return np.argmax(arr, axis=1)


def decode_labels(encoded_labels: ArrayLike, classes: Sequence) -> np.ndarray:
    """
    Convert integer label codes back into class names.

    Parameters
    ----------
    encoded_labels : array-like of shape (n_samples,)
        Integer-coded labels.

    classes : sequence
        Class names where classes[index] gives the decoded label.
    """
    labels = _as_1d_array(encoded_labels, name="encoded_labels").astype(int)
    classes_arr = np.asarray(classes, dtype=object)

    if classes_arr.ndim != 1 or classes_arr.size == 0:
        raise ValueError("classes must be a non-empty 1D sequence.")

    if np.any(labels < 0) or np.any(labels >= len(classes_arr)):
        raise ValueError("encoded_labels contains an index outside the classes array.")

    return classes_arr[labels]


def top_k_predictions(probabilities: ArrayLike, k: int = 3, classes: Optional[Sequence] = None):
    """
    Return the top-k predicted labels for each sample.

    Parameters
    ----------
    probabilities : array-like of shape (n_samples, n_classes)
        Class probability matrix.

    k : int
        Number of predictions to return per sample.

    classes : sequence, optional
        Class names. If omitted, class indices are returned.
    """
    probs = _as_2d_float(probabilities, name="probabilities")
    n_classes = probs.shape[1]

    if k < 1:
        raise ValueError("k must be at least 1.")

    k = min(k, n_classes)

    if classes is None:
        class_values = np.arange(n_classes, dtype=object)
    else:
        class_values = np.asarray(classes, dtype=object)
        if class_values.ndim != 1 or len(class_values) != n_classes:
            raise ValueError("classes must have one entry for each probability column.")

    order = np.argsort(probs, axis=1)[:, ::-1][:, :k]

    output = []
    for row_index in range(probs.shape[0]):
        row = []
        for class_index in order[row_index]:
            row.append((class_values[class_index], float(probs[row_index, class_index])))
        output.append(row)

    return output


def format_predictions(predictions: ArrayLike, probabilities: Optional[ArrayLike] = None, classes: Optional[Sequence] = None):
    """
    Format predictions into simple dictionaries.

    This is useful for readable notebook output without requiring pandas.

    Returns
    -------
    list of dict
        Each dictionary contains at least {"prediction": ...}. If probabilities
        are supplied, it also includes {"confidence": ...}.
    """
    preds = _as_1d_array(predictions, name="predictions")

    if classes is not None and np.issubdtype(preds.dtype, np.integer):
        display_predictions = decode_labels(preds.astype(int), classes)
    else:
        display_predictions = preds

    if probabilities is None:
        return [{"prediction": pred} for pred in display_predictions.tolist()]

    probs = np.asarray(probabilities, dtype=float)

    if probs.ndim == 1:
        if len(probs) != len(preds):
            raise ValueError("probabilities must have the same number of samples as predictions.")
        confidence = probs
    elif probs.ndim == 2:
        if probs.shape[0] != len(preds):
            raise ValueError("probabilities must have the same number of samples as predictions.")
        confidence = np.max(probs, axis=1)
    else:
        raise ValueError("probabilities must be a 1D or 2D array.")

    return [
        {"prediction": pred, "confidence": float(conf)}
        for pred, conf in zip(display_predictions.tolist(), confidence)
    ]
