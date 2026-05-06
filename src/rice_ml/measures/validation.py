"""
validation.py

Validation helpers for the rice_ml package.

This module provides simple tools for evaluating models without depending on
scikit-learn. The functions assume estimators follow the package's usual
interface: fit(X, y), predict(X), and optionally score(X, y).

Included tools
--------------
- k_fold_indices:
    Build reproducible K-fold train/test index splits.

- cross_val_score:
    Evaluate an estimator across K folds.

- stratified_split:
    Split a dataset while approximately preserving class proportions.

- holdout_evaluation:
    Train once on a train split and evaluate on a test split.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union
from copy import deepcopy
import numpy as np

try:
    from .metrics import accuracy_score
except ImportError:  # Allows direct script use if run outside the package.
    from metrics import accuracy_score

ArrayLike = Union[np.ndarray, Sequence, Sequence[Sequence]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Convert X into a non-empty 2D float NumPy array."""
    arr = np.asarray(X, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    return arr


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Convert y into a non-empty 1D NumPy array."""
    arr = np.asarray(y)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


def k_fold_indices(n_samples: int, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
    """
    Return train/test indices for K-fold cross-validation.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_splits > n_samples:
        raise ValueError("n_splits cannot be greater than n_samples.")

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    start = 0

    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        start = stop

    return folds


def cross_val_score(
    estimator,
    X: ArrayLike,
    y: ArrayLike,
    n_splits: int = 5,
    scoring: Optional[Callable] = None,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate an estimator using K-fold cross-validation.

    Parameters
    ----------
    estimator
        Model object with fit and predict methods. If scoring is None and the
        estimator has score(X, y), that score method is used.

    scoring : callable, optional
        Function with signature scoring(y_true, y_pred). If omitted, this uses
        estimator.score(X, y) when available, otherwise accuracy_score.
    """
    X = _ensure_2d_float(X)
    y = _ensure_1d(y)

    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples.")

    folds = k_fold_indices(
        n_samples=X.shape[0],
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    scores = []

    for train_idx, test_idx in folds:
        model = deepcopy(estimator)
        model.fit(X[train_idx], y[train_idx])

        if scoring is not None:
            predictions = model.predict(X[test_idx])
            score_value = scoring(y[test_idx], predictions)
        elif hasattr(model, "score"):
            score_value = model.score(X[test_idx], y[test_idx])
        else:
            predictions = model.predict(X[test_idx])
            score_value = accuracy_score(y[test_idx], predictions)

        scores.append(float(score_value))

    return np.array(scores)


def stratified_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
):
    """
    Split X and y while approximately preserving class proportions in y.
    """
    X = _ensure_2d_float(X)
    y = _ensure_1d(y)

    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n_test = int(round(len(label_indices) * test_size))

        if len(label_indices) > 1:
            n_test = min(max(n_test, 1), len(label_indices) - 1)

        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def holdout_evaluation(
    estimator,
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    scoring: Optional[Callable] = None,
    random_state: Optional[int] = None,
    stratify: bool = False,
):
    """
    Train and evaluate a model using a single train/test split.

    Returns
    -------
    dict
        Contains the fitted model, score, predictions, and split data.
    """
    X = _ensure_2d_float(X)
    y = _ensure_1d(y)

    if stratify:
        X_train, X_test, y_train, y_test = stratified_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)

        n_test = int(round(X.shape[0] * test_size))
        n_test = min(max(n_test, 1), X.shape[0] - 1)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    model = deepcopy(estimator)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if scoring is not None:
        score_value = scoring(y_test, predictions)
    elif hasattr(model, "score"):
        score_value = model.score(X_test, y_test)
    else:
        score_value = accuracy_score(y_test, predictions)

    return {
        "model": model,
        "score": float(score_value),
        "predictions": predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
