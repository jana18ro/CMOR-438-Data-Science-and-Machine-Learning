"""
validation.py

Validation helpers for the rice_ml package.

All functions assume estimators follow the package interface:
    fit(X, y)  →  self
    predict(X) →  np.ndarray
    score(X, y)  [optional]

Included tools
--------------
- k_fold_indices     : Build reproducible K-fold train/test index splits.
- cross_val_score    : Evaluate an estimator across K folds.
- stratified_split   : Train/test split that preserves class proportions.
- holdout_evaluation : Train once and evaluate on a held-out test set.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Sequence, Union

import numpy as np

try:
    from .metrics import accuracy_score
except ImportError:
    from metrics import accuracy_score  # type: ignore[no-redef]

ArrayLike = Union[np.ndarray, Sequence]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Return a non-empty 2-D float array (1-D input is treated as one column)."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return arr


def _to_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Return a non-empty 1-D array."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _check_X_y(X: ArrayLike, y: ArrayLike):
    """Validate and return a (2-D float X, 1-D y) pair with matching lengths."""
    Xa = _to_2d_float(X)
    ya = _to_1d(y)
    if Xa.shape[0] != len(ya):
        raise ValueError(
            f"X and y must have the same number of samples "
            f"(got {Xa.shape[0]} and {len(ya)})."
        )
    return Xa, ya


# ---------------------------------------------------------------------------
# K-fold indices
# ---------------------------------------------------------------------------


def k_fold_indices(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return train/test index pairs for K-fold cross-validation.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int
        Number of folds (≥ 2 and ≤ n_samples).
    shuffle : bool
        Whether to shuffle sample indices before splitting.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    list of (train_indices, test_indices) tuples
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed n_samples.")

    indices = np.arange(n_samples)
    if shuffle:
        np.random.default_rng(random_state).shuffle(indices)

    # Distribute remainder samples across the first folds
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        test_idx  = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        start = stop

    return folds


# ---------------------------------------------------------------------------
# Cross-validation score
# ---------------------------------------------------------------------------


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
        Model with ``fit`` and ``predict`` methods.  If *scoring* is ``None``
        and the estimator exposes a ``score(X, y)`` method that method is used;
        otherwise ``accuracy_score`` is used.
    scoring : callable(y_true, y_pred) → float, optional
        Custom scoring function.

    Returns
    -------
    np.ndarray
        Per-fold scores.
    """
    Xa, ya = _check_X_y(X, y)
    folds = k_fold_indices(
        n_samples=Xa.shape[0],
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    scores = []
    for train_idx, test_idx in folds:
        model = deepcopy(estimator)
        model.fit(Xa[train_idx], ya[train_idx])

        if scoring is not None:
            score_val = scoring(ya[test_idx], model.predict(Xa[test_idx]))
        elif hasattr(model, "score"):
            score_val = model.score(Xa[test_idx], ya[test_idx])
        else:
            score_val = accuracy_score(ya[test_idx], model.predict(Xa[test_idx]))

        scores.append(float(score_val))

    return np.array(scores)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X and y while approximately preserving class proportions.

    Parameters
    ----------
    test_size : float in (0, 1)
        Fraction of each class to place in the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    Xa, ya = _check_X_y(X, y)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")

    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx:  list[int] = []

    for label in np.unique(ya):
        cls_idx = np.where(ya == label)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, min(int(round(len(cls_idx) * test_size)), len(cls_idx) - 1))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())

    tr = np.array(train_idx)
    te = np.array(test_idx)
    rng.shuffle(tr)
    rng.shuffle(te)

    return Xa[tr], Xa[te], ya[tr], ya[te]


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------


def holdout_evaluation(
    estimator,
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    scoring: Optional[Callable] = None,
    random_state: Optional[int] = None,
    stratify: bool = False,
) -> dict:
    """
    Train and evaluate a model on a single train/test split.

    Parameters
    ----------
    stratify : bool
        When ``True``, use :func:`stratified_split` to preserve class balance.

    Returns
    -------
    dict with keys:
        model, score, predictions, X_train, X_test, y_train, y_test
    """
    Xa, ya = _check_X_y(X, y)

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")

    if stratify:
        X_train, X_test, y_train, y_test = stratified_split(
            Xa, ya, test_size=test_size, random_state=random_state
        )
    else:
        rng = np.random.default_rng(random_state)
        idx = np.arange(Xa.shape[0])
        rng.shuffle(idx)
        n_test   = max(1, min(int(round(Xa.shape[0] * test_size)), Xa.shape[0] - 1))
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]
        X_train, X_test = Xa[train_idx], Xa[test_idx]
        y_train, y_test = ya[train_idx], ya[test_idx]

    model = deepcopy(estimator)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if scoring is not None:
        score_val = scoring(y_test, predictions)
    elif hasattr(model, "score"):
        score_val = model.score(X_test, y_test)
    else:
        score_val = accuracy_score(y_test, predictions)

    return {
        "model":       model,
        "score":       float(score_val),
        "predictions": predictions,
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,
        "y_test":      y_test,
    }
