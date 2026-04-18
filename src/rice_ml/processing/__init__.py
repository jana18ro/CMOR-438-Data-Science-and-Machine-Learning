 # FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\processing\__init__.py

"""
rice_ml.processing
==================
Data preprocessing and postprocessing utilities for the rice_ml pipeline.

This subpackage enforces the standard ML data-handling discipline:
scalers and encoders are *fit only on training data* and then applied
(transform-only) to validation and test sets, preventing data leakage.

Preprocessing (preprocess.py)
------------------------------
StandardScaler
    Zero-mean, unit-variance standardisation.
    ``fit(X_train)`` -> ``transform(X)``
    Stores ``mean_`` and ``std_`` from the training set.

MinMaxScaler
    Scales each feature to the range [0, 1] (or a custom range).
    ``fit(X_train)`` -> ``transform(X)``
    Stores ``min_`` and ``scale_`` from the training set.

OrdinalEncoder
    Converts categorical string features to integer codes.
    ``fit(X_train)`` -> ``transform(X)``
    Stores ``categories_`` mapping from the training set.

train_test_split(X, y, test_size, random_state)
    Splits arrays into random train and test subsets.
    Returns: X_train, X_test, y_train, y_test

Postprocessing (postprocess.py)
---------------------------------
threshold_predictions(probs, threshold)
    Converts continuous probability outputs to binary labels
    using a configurable decision threshold (default 0.5).

decode_labels(encoded, mapping)
    Reverses label encoding back to original category strings.

format_predictions(y_pred, index)
    Wraps predictions in a pandas Series with a provided index
    for clean output and downstream compatibility.

Examples
--------
>>> from rice_ml.processing import StandardScaler, train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=42
... )
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train)
>>> X_test_scaled  = scaler.transform(X_test)   # uses training mean/std only

>>> from rice_ml.processing import MinMaxScaler
>>> mm = MinMaxScaler()
>>> X_norm = mm.fit_transform(X_train)
"""

from .preprocess import StandardScaler, MinMaxScaler, OrdinalEncoder, train_test_split
from .postprocess import threshold_predictions, decode_labels, format_predictions

__all__ = [
    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "OrdinalEncoder",
    "train_test_split",
    # Postprocessing
    "threshold_predictions",
    "decode_labels",
    "format_predictions",
]
