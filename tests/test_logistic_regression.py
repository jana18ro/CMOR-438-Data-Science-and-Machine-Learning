"""
test_logistic_regression.py

Comprehensive tests for LogisticRegression (binary classifier).

Covers:
- Attribute initialisation
- fit() convergence and parameter shapes
- predict_proba() range [0,1]
- predict() threshold at 0.5
- score() accuracy
- Sigmoid boundary behaviour
- Linearly separable data → high accuracy
- Non-trivial dataset (Breast Cancer)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/Jana CMOR/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def linearly_separable():
    """Two clearly separated 2-D blobs."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(50, 2))
    X1 = rng.normal(loc=[ 3,  3], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y

@pytest.fixture
def breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

@pytest.fixture
def fitted_sep(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression(learning_rate=0.5, n_iterations=500)
    m.fit(X, y)
    return m, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_lr(self):
        m = LogisticRegression()
        assert m.learning_rate == 0.01

    def test_default_iters(self):
        m = LogisticRegression()
        assert m.n_iterations == 1000

    def test_weights_none_before_fit(self):
        m = LogisticRegression()
        assert m.weights is None

    def test_bias_none_before_fit(self):
        m = LogisticRegression()
        assert m.bias is None

    def test_custom_params(self):
        m = LogisticRegression(learning_rate=0.5, n_iterations=200)
        assert m.learning_rate == 0.5
        assert m.n_iterations == 200


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit() structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_sets_weights(self, fitted_sep):
        m, X, y = fitted_sep
        assert m.weights is not None

    def test_fit_weights_shape(self, fitted_sep):
        m, X, y = fitted_sep
        assert m.weights.shape == (X.shape[1],)

    def test_fit_sets_bias(self, fitted_sep):
        m, X, y = fitted_sep
        assert m.bias is not None

    def test_fit_bias_is_scalar(self, fitted_sep):
        m, X, y = fitted_sep
        assert np.isscalar(m.bias) or m.bias.shape == ()

    def test_fit_1d_feature(self):
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0, 0, 1, 1])
        m = LogisticRegression(learning_rate=0.1, n_iterations=200)
        m.fit(X, y)
        assert m.weights.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# 3. predict_proba()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictProba:
    def test_proba_in_0_1(self, fitted_sep):
        m, X, y = fitted_sep
        proba = m.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_proba_shape(self, fitted_sep):
        m, X, y = fitted_sep
        proba = m.predict_proba(X)
        assert proba.shape == (len(X),)

    def test_proba_high_for_class1_region(self, linearly_separable):
        X, y = linearly_separable
        m = LogisticRegression(learning_rate=0.5, n_iterations=500)
        m.fit(X, y)
        # Points clearly in class-1 region should have high probability
        X_c1 = np.array([[3.0, 3.0], [3.5, 3.5]])
        proba = m.predict_proba(X_c1)
        assert np.all(proba > 0.5)

    def test_proba_low_for_class0_region(self, linearly_separable):
        X, y = linearly_separable
        m = LogisticRegression(learning_rate=0.5, n_iterations=500)
        m.fit(X, y)
        X_c0 = np.array([[-3.0, -3.0], [-3.5, -3.5]])
        proba = m.predict_proba(X_c0)
        assert np.all(proba < 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_binary_labels(self, fitted_sep):
        m, X, y = fitted_sep
        preds = m.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_shape(self, fitted_sep):
        m, X, y = fitted_sep
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_consistent_with_proba(self, fitted_sep):
        m, X, y = fitted_sep
        proba = m.predict_proba(X)
        preds = m.predict(X)
        expected = (proba >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, expected)

    def test_predict_correct_on_separable(self, linearly_separable):
        X, y = linearly_separable
        m = LogisticRegression(learning_rate=0.5, n_iterations=500)
        m.fit(X, y)
        preds = m.predict(X)
        assert np.mean(preds == y) > 0.95

    def test_predict_dtype_int(self, fitted_sep):
        m, X, y = fitted_sep
        preds = m.predict(X)
        assert preds.dtype in [np.int32, np.int64, int] or np.issubdtype(preds.dtype, np.integer)


# ─────────────────────────────────────────────────────────────────────────────
# 5. score()
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_between_0_and_1(self, fitted_sep):
        m, X, y = fitted_sep
        s = m.score(X, y)
        assert 0 <= s <= 1

    def test_score_high_on_separable(self, linearly_separable):
        X, y = linearly_separable
        m = LogisticRegression(learning_rate=0.5, n_iterations=500)
        m.fit(X, y)
        assert m.score(X, y) > 0.95

    def test_score_returns_float(self, fitted_sep):
        m, X, y = fitted_sep
        assert isinstance(m.score(X[: 10], y[:10]), float)

    def test_score_equal_to_manual_accuracy(self, fitted_sep):
        m, X, y = fitted_sep
        preds = m.predict(X)
        manual = np.mean(preds == y)
        assert abs(m.score(X, y) - manual) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration — Breast Cancer
# ─────────────────────────────────────────────────────────────────────────────

class TestBreastCancer:
    def test_accuracy_above_95(self, breast_cancer_data):
        X_train, X_test, y_train, y_test = breast_cancer_data
        m = LogisticRegression(learning_rate=0.1, n_iterations=500)
        m.fit(X_train, y_train)
        assert m.score(X_test, y_test) > 0.95

    def test_weights_shape_30_features(self, breast_cancer_data):
        X_train, X_test, y_train, y_test = breast_cancer_data
        m = LogisticRegression(learning_rate=0.1, n_iterations=200)
        m.fit(X_train, y_train)
        assert m.weights.shape == (30,)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sigmoid stability
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmoidStability:
    def test_no_nan_in_proba(self):
        """Large positive/negative inputs must not produce NaN."""
        X = np.array([[1000.0], [-1000.0], [0.0]])
        y = np.array([1, 0, 1])
        m = LogisticRegression(learning_rate=0.01, n_iterations=1)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert not np.any(np.isnan(proba))

    def test_proba_clipped_in_0_1(self):
        X = np.array([[500.0], [-500.0]])
        y = np.array([1, 0])
        m = LogisticRegression(learning_rate=0.01, n_iterations=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)
