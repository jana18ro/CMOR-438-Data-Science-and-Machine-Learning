"""
test_perceptron.py
==================
Comprehensive pytest test suite for the ``Perceptron`` class defined in
``perceptron.py``.

Test Structure
--------------
Tests are grouped into four logical sections:

1. **Initialisation tests** — verify default and custom constructor parameters,
   and confirm attributes are None / empty before fitting.

2. **Fitting & prediction tests** — verify that the model trains correctly
   on controlled synthetic datasets (AND gate, OR gate, linearly separable
   blobs) and that predictions have the right dtype and shape.

3. **Metric & utility tests** — verify accuracy, confusion matrix, loss
   history, plot_loss, get_params, and __repr__.

4. **Edge case & error tests** — verify graceful failures on unfitted models,
   mismatched shapes, single-sample data, all-zero features, and
   already-perfectly-labelled data.

Running the tests
-----------------
From the project root directory::

    pytest test_perceptron.py -v

Or with coverage::

    pytest test_perceptron.py -v --cov=perceptron --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for CI

from rice_ml.supervised_learning.perceptron import Perceptron


# ===========================================================================
# Fixtures — reusable data shared across tests
# ===========================================================================

@pytest.fixture
def and_gate():
    """
    AND gate dataset — canonical linearly separable 2-D binary problem.

    Only the input (1, 1) should yield label 1; all others yield 0.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    return X, y


@pytest.fixture
def or_gate():
    """
    OR gate dataset — linearly separable; label is 1 unless both inputs are 0.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    return X, y


@pytest.fixture
def blob_dataset():
    """
    Larger linearly separable synthetic blob dataset (200 samples, 4 features).

    Generated with a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-2.0, scale=0.8, size=(100, 4))
    X1 = rng.normal(loc=+2.0, scale=0.8, size=(100, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def fitted_and(and_gate):
    """
    A Perceptron already fitted on the AND gate — shared across metric tests.
    """
    X, y = and_gate
    clf = Perceptron(learning_rate=0.1, n_iterations=20, random_state=0)
    clf.fit(X, y)
    return clf, X, y


# ===========================================================================
# 1. Initialisation tests
# ===========================================================================

class TestInit:
    """Tests covering constructor defaults and attribute pre-fit state."""

    def test_default_learning_rate(self):
        """Default learning_rate should be 0.01."""
        clf = Perceptron()
        assert clf.learning_rate == 0.01, (
            f"Expected 0.01, got {clf.learning_rate}"
        )

    def test_default_n_iterations(self):
        """Default n_iterations should be 1000."""
        clf = Perceptron()
        assert clf.n_iterations == 1000

    def test_default_random_state(self):
        """Default random_state should be 42."""
        clf = Perceptron()
        assert clf.random_state == 42

    def test_custom_params(self):
        """Custom constructor values should be stored correctly."""
        clf = Perceptron(learning_rate=0.5, n_iterations=50, random_state=7)
        assert clf.learning_rate == 0.5
        assert clf.n_iterations == 50
        assert clf.random_state == 7

    def test_weights_none_before_fit(self):
        """weights_ must be None before fit is called."""
        clf = Perceptron()
        assert clf.weights_ is None

    def test_bias_none_before_fit(self):
        """bias_ must be None before fit is called."""
        clf = Perceptron()
        assert clf.bias_ is None

    def test_loss_empty_before_fit(self):
        """loss_ must be an empty list before fit is called."""
        clf = Perceptron()
        assert clf.loss_ == []

    def test_repr(self):
        """__repr__ should return a string containing key param names."""
        clf = Perceptron(learning_rate=0.05, n_iterations=200, random_state=1)
        r = repr(clf)
        assert "learning_rate=0.05" in r
        assert "n_iterations=200" in r
        assert "random_state=1" in r


# ===========================================================================
# 2. Fitting & Prediction tests
# ===========================================================================

class TestFitPredict:
    """Tests covering the training loop and prediction output."""

    def test_fit_returns_self(self, and_gate):
        """fit() must return the Perceptron instance (for method chaining)."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=10, random_state=0)
        result = clf.fit(X, y)
        assert result is clf

    def test_weights_initialised_after_fit(self, and_gate):
        """weights_ should be a 1-D array with length == n_features after fit."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=10, random_state=0)
        clf.fit(X, y)
        assert isinstance(clf.weights_, np.ndarray)
        assert clf.weights_.shape == (X.shape[1],)

    def test_bias_is_float_after_fit(self, and_gate):
        """bias_ should be a Python float (or float64) after fit."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=10, random_state=0)
        clf.fit(X, y)
        assert isinstance(clf.bias_, (float, np.floating))

    def test_n_features_stored(self, blob_dataset):
        """n_features_in_ should equal the number of columns in X."""
        X, y = blob_dataset
        clf = Perceptron(learning_rate=0.1, n_iterations=30, random_state=0)
        clf.fit(X, y)
        assert clf.n_features_in_ == X.shape[1]

    def test_predict_output_shape(self, and_gate):
        """predict() output shape must match (n_samples,)."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=20, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape, (
            f"Expected shape {y.shape}, got {preds.shape}"
        )

    def test_predict_binary_values(self, and_gate):
        """predict() must return only 0s and 1s."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=20, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        unique = set(np.unique(preds))
        assert unique.issubset({0, 1}), (
            f"Unexpected prediction values: {unique}"
        )

    def test_and_gate_accuracy(self, and_gate):
        """
        Perceptron should achieve >= 75% accuracy on the AND gate.

        The AND gate is linearly separable; with a reasonable lr and sufficient
        iterations it should achieve perfect or near-perfect accuracy.
        """
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=50, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        assert acc >= 0.75, f"AND gate accuracy too low: {acc:.2%}"

    def test_or_gate_accuracy(self, or_gate):
        """Perceptron should achieve >= 75% accuracy on the OR gate."""
        X, y = or_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=50, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        assert acc >= 0.75, f"OR gate accuracy too low: {acc:.2%}"

    def test_blob_dataset_high_accuracy(self, blob_dataset):
        """
        On a well-separated linearly separable dataset the Perceptron should
        achieve >= 90% accuracy.
        """
        X, y = blob_dataset
        clf = Perceptron(learning_rate=0.1, n_iterations=100, random_state=0)
        clf.fit(X, y)
        acc = clf.accuracy(X, y)
        assert acc >= 0.90, f"Blob accuracy too low: {acc:.2%}"

    def test_reproducibility_with_same_seed(self, blob_dataset):
        """
        Two Perceptrons with the same random_state must produce identical
        predictions after identical training.
        """
        X, y = blob_dataset
        clf1 = Perceptron(learning_rate=0.05, n_iterations=50, random_state=99)
        clf2 = Perceptron(learning_rate=0.05, n_iterations=50, random_state=99)
        clf1.fit(X, y)
        clf2.fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_different_seeds_may_differ(self, blob_dataset):
        """
        Different random seeds should (in general) yield different initial
        weights; verify that at least the weights themselves differ.
        """
        X, y = blob_dataset
        clf1 = Perceptron(learning_rate=0.05, n_iterations=1, random_state=1)
        clf2 = Perceptron(learning_rate=0.05, n_iterations=1, random_state=2)
        clf1.fit(X, y)
        clf2.fit(X, y)
        # Initial weights are drawn differently, so they should differ
        assert not np.array_equal(clf1.weights_, clf2.weights_)

    def test_method_chaining(self, and_gate):
        """fit().predict() chain should work without intermediate variable."""
        X, y = and_gate
        preds = Perceptron(
            learning_rate=0.1, n_iterations=30, random_state=0
        ).fit(X, y).predict(X)
        assert preds.shape == y.shape


# ===========================================================================
# 3. Metrics & utility tests
# ===========================================================================

class TestMetrics:
    """Tests covering accuracy, confusion matrix, loss, and helpers."""

    def test_accuracy_perfect(self):
        """
        On linearly separable data with enough iterations accuracy should
        reach 1.0 (100%).
        """
        rng = np.random.default_rng(7)
        X0 = rng.normal(-3, 0.5, (50, 2))
        X1 = rng.normal(+3, 0.5, (50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)
        clf = Perceptron(learning_rate=0.1, n_iterations=200, random_state=0)
        clf.fit(X, y)
        assert clf.accuracy(X, y) == 1.0

    def test_accuracy_range(self, fitted_and):
        """accuracy() must always return a value in [0.0, 1.0]."""
        clf, X, y = fitted_and
        acc = clf.accuracy(X, y)
        assert 0.0 <= acc <= 1.0

    def test_loss_length(self, and_gate):
        """loss_ should have exactly n_iterations entries after fit."""
        X, y = and_gate
        n_iter = 25
        clf = Perceptron(learning_rate=0.1, n_iterations=n_iter, random_state=0)
        clf.fit(X, y)
        assert len(clf.loss_) == n_iter, (
            f"Expected {n_iter} loss entries, got {len(clf.loss_)}"
        )

    def test_loss_values_non_negative(self, and_gate):
        """All MSE loss values must be >= 0."""
        X, y = and_gate
        clf = Perceptron(learning_rate=0.1, n_iterations=20, random_state=0)
        clf.fit(X, y)
        assert all(l >= 0 for l in clf.loss_), "Negative MSE found in loss_"

    def test_loss_decreases_on_separable_data(self, blob_dataset):
        """
        On well-separated data the final loss should be less than or equal to
        the initial loss (general downward trend expected).
        """
        X, y = blob_dataset
        clf = Perceptron(learning_rate=0.1, n_iterations=100, random_state=0)
        clf.fit(X, y)
        assert clf.loss_[-1] <= clf.loss_[0], (
            "Loss did not decrease over training on linearly separable data."
        )

    def test_confusion_matrix_returns_dataframe(self, fitted_and):
        """confusion_matrix() must return a pandas DataFrame."""
        clf, X, y = fitted_and
        cm = clf.confusion_matrix(X, y)
        assert isinstance(cm, pd.DataFrame)

    def test_confusion_matrix_sums_to_n_samples(self, fitted_and):
        """All confusion matrix entries must sum to the number of samples."""
        clf, X, y = fitted_and
        cm = clf.confusion_matrix(X, y)
        assert cm.values.sum() == len(y), (
            f"CM sum {cm.values.sum()} != n_samples {len(y)}"
        )

    def test_confusion_matrix_non_negative(self, fitted_and):
        """Confusion matrix values must all be non-negative integers."""
        clf, X, y = fitted_and
        cm = clf.confusion_matrix(X, y)
        assert (cm.values >= 0).all()

    def test_get_params(self):
        """get_params() should return a dict with the three hyperparameters."""
        clf = Perceptron(learning_rate=0.2, n_iterations=42, random_state=3)
        params = clf.get_params()
        assert isinstance(params, dict)
        assert params["learning_rate"] == 0.2
        assert params["n_iterations"] == 42
        assert params["random_state"] == 3

    def test_plot_loss_no_error(self, fitted_and, tmp_path):
        """plot_loss() should run without raising and save a PNG file."""
        clf, _, _ = fitted_and
        out = tmp_path / "loss.png"
        clf.plot_loss(save_path=str(out))   # save_path avoids plt.show()
        assert out.exists(), "plot_loss() did not create the expected file."

    def test_plot_loss_before_fit_raises(self):
        """plot_loss() before fit should raise RuntimeError."""
        clf = Perceptron()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            clf.plot_loss()


# ===========================================================================
# 4. Edge case & error tests
# ===========================================================================

class TestEdgeCases:
    """Tests for boundary conditions and expected failures."""

    def test_predict_before_fit_raises_typeerror(self):
        """
        Calling predict() on an unfitted model should raise TypeError because
        weights_ is None and np.dot cannot operate on it.
        """
        clf = Perceptron()
        X = np.random.rand(5, 3)
        with pytest.raises(TypeError):
            clf.predict(X)

    def test_accuracy_before_fit_raises(self):
        """accuracy() before fit should also fail (weights_ is None)."""
        clf = Perceptron()
        X = np.random.rand(5, 3)
        y = np.zeros(5, dtype=int)
        with pytest.raises((TypeError, AttributeError)):
            clf.accuracy(X, y)

    def test_mismatched_X_y_raises_valueerror(self):
        """fit() with X.shape[0] != y.shape[0] should raise ValueError."""
        clf = Perceptron()
        X = np.random.rand(10, 2)
        y = np.zeros(8, dtype=int)        # wrong number of labels
        with pytest.raises(ValueError, match="samples"):
            clf.fit(X, y)

    def test_single_sample_fit(self):
        """
        fit() on a single-sample dataset should not raise and should produce
        valid (though potentially trivial) predictions.
        """
        X = np.array([[1.0, 2.0]])
        y = np.array([1])
        clf = Perceptron(learning_rate=0.1, n_iterations=5, random_state=0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (1,)
        assert pred[0] in (0, 1)

    def test_all_zero_features(self):
        """
        When all features are zero the weights cannot update (delta * 0 = 0).
        The bias may still update.  fit() should complete without error.
        """
        X = np.zeros((6, 3))
        y = np.array([0, 1, 0, 1, 0, 1])
        clf = Perceptron(learning_rate=0.1, n_iterations=10, random_state=0)
        clf.fit(X, y)                      # should not raise
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_all_same_label(self):
        """
        If all labels are the same class, the model should fit without error
        and predict consistently.
        """
        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 2))
        y = np.ones(20, dtype=int)          # all label-1
        clf = Perceptron(learning_rate=0.1, n_iterations=20, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_high_dimensional_input(self):
        """
        Perceptron should handle high-dimensional inputs (e.g. 500 features)
        without error.  Validates NumPy operations scale correctly.
        """
        rng = np.random.default_rng(8)
        X = rng.standard_normal((50, 500))
        y = (X[:, 0] > 0).astype(int)      # label based on first feature
        clf = Perceptron(learning_rate=0.01, n_iterations=20, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_large_learning_rate_does_not_crash(self):
        """
        Even with a very large learning rate (causing oscillation) the
        Perceptron should complete training without numerical errors.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        clf = Perceptron(learning_rate=10.0, n_iterations=30, random_state=0)
        clf.fit(X, y)                      # should not raise
        clf.predict(X)

    def test_zero_iterations(self):
        """
        With n_iterations=0 the weights should remain at their initial random
        values (no updates), and loss_ should be empty.
        """
        X = np.array([[0, 0], [1, 1]], dtype=float)
        y = np.array([0, 1])
        clf = Perceptron(learning_rate=0.1, n_iterations=0, random_state=0)
        clf.fit(X, y)
        assert clf.loss_ == []
        # weights exist (initialised) but are unchanged
        assert clf.weights_ is not None

    def test_predict_shape_matches_input_rows(self, blob_dataset):
        """
        predict() output length must always match the number of rows in X,
        including subsets of the training data.
        """
        X, y = blob_dataset
        clf = Perceptron(learning_rate=0.1, n_iterations=50, random_state=0)
        clf.fit(X, y)
        for n in [1, 10, 50, 200]:
            preds = clf.predict(X[:n])
            assert len(preds) == n, f"Expected {n} preds, got {len(preds)}"
