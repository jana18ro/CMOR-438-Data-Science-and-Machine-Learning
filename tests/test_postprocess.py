"""
test_postprocess.py

Tests for rice_ml.processing.postprocess.

These tests cover thresholding, label conversion, decoding, top-k output, and
formatted predictions.
"""

import numpy as np
import pytest

from rice_ml.processing.postprocess import (
    apply_threshold,
    one_hot_to_labels,
    decode_labels,
    top_k_predictions,
    format_predictions,
)


def test_apply_threshold_default_labels():
    probabilities = np.array([0.2, 0.5, 0.8])

    labels = apply_threshold(probabilities, threshold=0.5)

    assert np.array_equal(labels, np.array([0, 1, 1]))


def test_apply_threshold_custom_labels():
    probabilities = np.array([0.1, 0.9])

    labels = apply_threshold(
        probabilities,
        threshold=0.7,
        positive_label="yes",
        negative_label="no",
    )

    assert np.array_equal(labels, np.array(["no", "yes"]))


def test_apply_threshold_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.1, 0.2]), threshold=1.5)


def test_one_hot_to_labels_works_for_one_hot_and_probabilities():
    values = np.array([
        [1.0, 0.0, 0.0],
        [0.1, 0.7, 0.2],
        [0.2, 0.3, 0.5],
    ])

    labels = one_hot_to_labels(values)

    assert np.array_equal(labels, np.array([0, 1, 2]))


def test_decode_labels_converts_indices_to_class_names():
    encoded = np.array([0, 2, 1])
    classes = np.array(["setosa", "versicolor", "virginica"])

    decoded = decode_labels(encoded, classes)

    assert np.array_equal(decoded, np.array(["setosa", "virginica", "versicolor"]))


def test_decode_labels_rejects_invalid_indices():
    encoded = np.array([0, 3])
    classes = np.array(["a", "b"])

    with pytest.raises(ValueError):
        decode_labels(encoded, classes)


def test_top_k_predictions_returns_ordered_pairs_with_indices():
    probabilities = np.array([
        [0.1, 0.8, 0.1],
        [0.4, 0.2, 0.4],
    ])

    top = top_k_predictions(probabilities, k=2)

    assert len(top) == 2
    assert len(top[0]) == 2
    assert top[0][0][0] == 1
    assert top[0][0][1] == pytest.approx(0.8)


def test_top_k_predictions_uses_class_names():
    probabilities = np.array([
        [0.1, 0.8, 0.1],
    ])
    classes = ["cat", "dog", "bird"]

    top = top_k_predictions(probabilities, k=2, classes=classes)

    assert top[0][0][0] == "dog"
    assert top[0][0][1] == pytest.approx(0.8)


def test_top_k_predictions_caps_k_at_number_of_classes():
    probabilities = np.array([
        [0.2, 0.8],
    ])

    top = top_k_predictions(probabilities, k=10)

    assert len(top[0]) == 2


def test_format_predictions_without_probabilities():
    predictions = np.array(["spam", "not spam"])

    formatted = format_predictions(predictions)

    assert formatted == [
        {"prediction": "spam"},
        {"prediction": "not spam"},
    ]


def test_format_predictions_with_1d_confidence_values():
    predictions = np.array(["yes", "no"])
    confidence = np.array([0.9, 0.7])

    formatted = format_predictions(predictions, probabilities=confidence)

    assert formatted[0]["prediction"] == "yes"
    assert formatted[0]["confidence"] == pytest.approx(0.9)
    assert formatted[1]["prediction"] == "no"
    assert formatted[1]["confidence"] == pytest.approx(0.7)


def test_format_predictions_with_2d_probabilities_uses_max_confidence():
    predictions = np.array([1, 0])
    probabilities = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ])

    formatted = format_predictions(predictions, probabilities=probabilities)

    assert formatted[0]["confidence"] == pytest.approx(0.8)
    assert formatted[1]["confidence"] == pytest.approx(0.7)


def test_format_predictions_decodes_integer_predictions_when_classes_given():
    predictions = np.array([1, 0])
    probabilities = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ])
    classes = ["negative", "positive"]

    formatted = format_predictions(predictions, probabilities=probabilities, classes=classes)

    assert formatted[0]["prediction"] == "positive"
    assert formatted[1]["prediction"] == "negative"


def test_postprocess_shape_errors():
    with pytest.raises(ValueError):
        one_hot_to_labels(np.array([1, 0, 0]))

    with pytest.raises(ValueError):
        top_k_predictions(np.array([0.2, 0.8]), k=1)

    with pytest.raises(ValueError):
        format_predictions(np.array([0, 1]), probabilities=np.array([0.5]))
