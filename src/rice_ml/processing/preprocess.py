"""
preprocess.py

Preprocessing utilities for the rice_ml package.

This module provides small, from-scratch tools for preparing data before it is
passed into a machine learning model. The goal is to keep preprocessing clear,
testable, and consistent with the rest of the package's NumPy-based style.

Included tools
--------------
- StandardScaler:
    Standardizes features by subtracting the training mean and dividing by the
    training standard deviation.

- MinMaxScaler:
    Scales each feature into a chosen numeric range.

- OrdinalEncoder:
    Converts categorical labels or feature columns into integer codes.

- train_test_split:
    Splits arrays into training and testing sets with optional shuffling and
    optional stratification.

The scalers follow the standard fit-on-train, transform-on-test workflow.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence, Sequence[Sequence]]


def _ensure_2d_array(X: ArrayLike, *, dtype=float, name: str = "X") -> np.ndarray:
    """Convert X into a non-empty 2D NumPy array."""
    arr = np.asarray(X, dtype=dtype)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    return arr


def _ensure_1d_array(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Convert y into a non-empty 1D NumPy array."""
    arr = np.asarray(y)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")

    return arr


class StandardScaler:
    """
    Standardize features using the training mean and standard deviation.

    Each feature is transformed as:

        z = (x - mean) / standard_deviation

    Constant columns are handled safely by using a scale value of 1.0.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X: ArrayLike) -> "StandardScaler":
        """Learn feature-wise mean and standard deviation from X."""
        X = _ensure_2d_array(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply the learned standardization to X."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")

        X = _ensure_2d_array(X, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as the fitted data.")

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit the scaler and return the transformed data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Undo the standardization."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform.")

        X = _ensure_2d_array(X, dtype=float)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Scale each feature into a chosen range.

    By default, values are scaled into the interval [0, 1].
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must contain exactly two values.")

        low, high = float(feature_range[0]), float(feature_range[1])

        if low >= high:
            raise ValueError("feature_range must be ordered as (min, max).")

        self.feature_range = (low, high)
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X: ArrayLike) -> "MinMaxScaler":
        """Learn feature-wise minimum and maximum values from X."""
        X = _ensure_2d_array(X, dtype=float)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0

        self.scale_ = data_range
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Scale X using the fitted feature ranges."""
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform.")

        X = _ensure_2d_array(X, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as the fitted data.")

        low, high = self.feature_range
        X_zero_one = (X - self.data_min_) / self.scale_

        return X_zero_one * (high - low) + low

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit the scaler and return the transformed data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Undo the min-max scaling."""
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse_transform.")

        X = _ensure_2d_array(X, dtype=float)
        low, high = self.feature_range
        X_zero_one = (X - low) / (high - low)

        return X_zero_one * self.scale_ + self.data_min_


class OrdinalEncoder:
    """
    Encode categorical values as integer codes.

    The encoder can handle a 1D label vector or a 2D categorical feature matrix.
    Each column is encoded independently.
    """

    def __init__(self):
        self.categories_ = None
        self.category_maps_ = None
        self.n_features_in_ = None
        self._input_was_1d = False

    def fit(self, X: ArrayLike) -> "OrdinalEncoder":
        """Learn the category-to-integer mapping."""
        arr = np.asarray(X, dtype=object)

        if arr.ndim == 1:
            self._input_was_1d = True
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            self._input_was_1d = False
        else:
            raise ValueError("X must be a 1D or 2D array.")

        self.categories_ = []
        self.category_maps_ = []

        for col in range(arr.shape[1]):
            categories = np.unique(arr[:, col])
            mapping = {value: index for index, value in enumerate(categories)}

            self.categories_.append(categories)
            self.category_maps_.append(mapping)

        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Convert categories into integer codes."""
        if self.category_maps_ is None:
            raise RuntimeError("OrdinalEncoder must be fitted before transform.")

        arr = np.asarray(X, dtype=object)

        input_was_1d = arr.ndim == 1

        if input_was_1d:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if arr.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of columns as the fitted data.")

        encoded = np.zeros(arr.shape, dtype=int)

        for col, mapping in enumerate(self.category_maps_):
            for row, value in enumerate(arr[:, col]):
                if value not in mapping:
                    raise ValueError(f"Unknown category {value!r} in column {col}.")
                encoded[row, col] = mapping[value]

        return encoded.ravel() if input_was_1d else encoded

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit the encoder and return encoded values."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Convert integer codes back into original categories."""
        if self.categories_ is None:
            raise RuntimeError("OrdinalEncoder must be fitted before inverse_transform.")

        arr = np.asarray(X, dtype=int)

        input_was_1d = arr.ndim == 1

        if input_was_1d:
            arr = arr.reshape(-1, 1)

        if arr.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of columns as the fitted data.")

        decoded = np.empty(arr.shape, dtype=object)

        for col, categories in enumerate(self.categories_):
            if np.any(arr[:, col] < 0) or np.any(arr[:, col] >= len(categories)):
                raise ValueError(f"Column {col} contains an invalid encoded value.")
            decoded[:, col] = categories[arr[:, col]]

        return decoded.ravel() if input_was_1d else decoded


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    test_size: float = 0.25,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratify: Optional[ArrayLike] = None,
):
    """
    Split arrays into train and test subsets.

    If y is provided, the same row indices are used for X and y. If stratify is
    provided, the split attempts to preserve class proportions in the test set.
    """
    X = _ensure_2d_array(X, dtype=float)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    y_arr = None if y is None else _ensure_1d_array(y)

    if y_arr is not None and len(y_arr) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    if stratify is not None:
        stratify_arr = _ensure_1d_array(stratify, name="stratify")

        if len(stratify_arr) != n_samples:
            raise ValueError("stratify must have the same number of samples as X.")

        train_indices = []
        test_indices = []

        for label in np.unique(stratify_arr):
            label_indices = np.where(stratify_arr == label)[0]

            if shuffle:
                rng.shuffle(label_indices)

            n_test_label = int(round(len(label_indices) * test_size))

            if len(label_indices) > 1:
                n_test_label = min(max(n_test_label, 1), len(label_indices) - 1)

            test_indices.extend(label_indices[:n_test_label])
            train_indices.extend(label_indices[n_test_label:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

    else:
        indices = np.arange(n_samples)

        if shuffle:
            rng.shuffle(indices)

        n_test = int(round(n_samples * test_size))
        n_test = min(max(n_test, 1), n_samples - 1)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    if y_arr is None:
        return X_train, X_test

    y_train = y_arr[train_indices]
    y_test = y_arr[test_indices]

    return X_train, X_test, y_train, y_test


# Functional aliases for quick use in notebooks.

def standardize(X: ArrayLike) -> np.ndarray:
    """Return a standardized copy of X."""
    return StandardScaler().fit_transform(X)


def minmax_scale(X: ArrayLike, feature_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Return a min-max scaled copy of X."""
    return MinMaxScaler(feature_range=feature_range).fit_transform(X)
