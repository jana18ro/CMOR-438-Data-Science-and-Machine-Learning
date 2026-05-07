"""
test_preprocess.py

Tests for rice_ml.processing.preprocess.

These tests cover scalers, ordinal encoding, train/test splitting, and basic
error handling.
"""

import numpy as np
import pytest

from rice_ml.processing.preprocess import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    train_test_split,
    standardize,
    minmax_scale,
)


def test_standard_scaler_fit_transform_centers_and_scales_columns():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0]))
    assert np.allclose(np.std(X_scaled, axis=0), np.array([1.0, 1.0]))


def test_standard_scaler_inverse_transform_recovers_original_data():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)

    assert np.allclose(X_recovered, X)


def test_standard_scaler_handles_constant_columns():
    X = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled[:, 1], np.zeros(3))
    assert scaler.scale_[1] == pytest.approx(1.0) # type: ignore


def test_minmax_scaler_default_range():
    X = np.array([
        [0.0, 10.0],
        [5.0, 20.0],
        [10.0, 30.0],
    ])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.min(X_scaled, axis=0), np.array([0.0, 0.0]))
    assert np.allclose(np.max(X_scaled, axis=0), np.array([1.0, 1.0]))


def test_minmax_scaler_custom_range_and_inverse_transform():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    assert np.min(X_scaled) == pytest.approx(-1.0)
    assert np.max(X_scaled) == pytest.approx(1.0)
    assert np.allclose(scaler.inverse_transform(X_scaled), X)


def test_ordinal_encoder_1d_labels_round_trip():
    y = np.array(["cat", "dog", "cat", "bird"])

    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(y)
    decoded = encoder.inverse_transform(encoded)

    assert encoded.ndim == 1
    assert np.array_equal(decoded, y)


def test_ordinal_encoder_2d_features_round_trip():
    X = np.array([
        ["red", "small"],
        ["blue", "large"],
        ["red", "large"],
    ], dtype=object)

    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(X)
    decoded = encoder.inverse_transform(encoded)

    assert encoded.shape == X.shape
    assert np.array_equal(decoded, X)


def test_ordinal_encoder_rejects_unknown_category():
    encoder = OrdinalEncoder()
    encoder.fit(np.array(["a", "b"]))

    with pytest.raises(ValueError):
        encoder.transform(np.array(["c"]))


def test_train_test_split_returns_aligned_arrays():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split( # type: ignore
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_train_test_split_without_y_returns_two_arrays():
    X = np.arange(12).reshape(6, 2)

    X_train, X_test = train_test_split(X, test_size=0.5, random_state=1) # type: ignore

    assert len(X_train) + len(X_test) == len(X)


def test_train_test_split_stratify_preserves_both_classes():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split( # type: ignore
        X,
        y,
        test_size=0.4,
        random_state=2,
        stratify=y,
    )

    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_functional_scaling_helpers():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    assert np.allclose(standardize(X), StandardScaler().fit_transform(X))
    assert np.allclose(minmax_scale(X), MinMaxScaler().fit_transform(X))


def test_scalers_raise_before_fit():
    X = np.array([[1.0], [2.0]])

    with pytest.raises(RuntimeError):
        StandardScaler().transform(X)

    with pytest.raises(RuntimeError):
        MinMaxScaler().transform(X)
