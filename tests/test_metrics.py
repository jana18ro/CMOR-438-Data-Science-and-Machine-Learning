"""
test_metrics.py

Tests for rice_ml.measures.metrics.

These tests cover classification metrics, regression metrics, aliases, and
basic error handling.
"""

import numpy as np
import pytest

from rice_ml.measures.metrics import (
    accuracy_score,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    mse,
    rmse,
    mae,
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
)


def test_accuracy_score_counts_correct_predictions():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)


def test_confusion_matrix_binary_labels():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    expected = np.array([
        [1, 1],
        [1, 1],
    ])

    assert np.array_equal(confusion_matrix(y_true, y_pred, labels=[0, 1]), expected)


def test_precision_recall_f1_binary():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0])

    assert precision(y_true, y_pred, positive_label=1) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred, positive_label=1) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred, positive_label=1) == pytest.approx(2 / 3)


def test_precision_recall_f1_macro_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])

    macro_precision = precision(y_true, y_pred, average="macro", labels=[0, 1, 2])
    macro_recall = recall(y_true, y_pred, average="macro", labels=[0, 1, 2])
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])

    assert 0 <= macro_precision <= 1
    assert 0 <= macro_recall <= 1
    assert 0 <= macro_f1 <= 1


def test_per_class_metrics_return_arrays_when_average_is_none():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    p = precision(y_true, y_pred, average=None, labels=[0, 1])
    r = recall(y_true, y_pred, average=None, labels=[0, 1])
    f = f1_score(y_true, y_pred, average=None, labels=[0, 1])

    assert isinstance(p, np.ndarray)
    assert isinstance(r, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert p.shape == (2,)
    assert r.shape == (2,)
    assert f.shape == (2,)


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    assert mse(y_true, y_pred) == pytest.approx(0.375)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert mae(y_true, y_pred) == pytest.approx(0.5)

    # Known sklearn-style R^2 result for this example.
    assert r2_score(y_true, y_pred) == pytest.approx(0.9486081370449679)


def test_metric_aliases_match_original_functions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(mse(y_true, y_pred))
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(rmse(y_true, y_pred))


def test_metrics_raise_on_mismatched_lengths():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])

    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)

    with pytest.raises(ValueError):
        mse(y_true, y_pred)


def test_binary_precision_requires_positive_label_for_multiclass():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])

    with pytest.raises(ValueError):
        precision(y_true, y_pred, average="binary")
