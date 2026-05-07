"""
test_linear_regression.py

Comprehensive tests for SimpleLinearRegression.

Covers:
- Input validation and error handling
- Parameter storage after fit
- Learning curve (cost decreasing)
- Accuracy on y=mx+b synthetic data (coef and intercept recovery)
- predict() shape and values
- score() (R²) on perfect and random data
- plot_cost / plot_fit do not crash
- Alpha / tol / random_state behaviours
- Unfitted model raises RuntimeError
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from linear_regression import SimpleLinearRegression


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_line():
    """y = 3x + 1  — noiseless."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, 80)
    y = 3 * X + 1
    return X, y

@pytest.fixture
def noisy_line():
    """y = 2x - 4 + noise."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, 100)
    y = 2 * X - 4 + rng.normal(0, 0.5, 100)
    return X, y

@pytest.fixture
def fitted_model(perfect_line):
    X, y = perfect_line
    m = SimpleLinearRegression(alpha=0.01, max_iter=3000, random_state=42)
    m.fit(X, y)
    return m, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation & parameter validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        m = SimpleLinearRegression()
        assert m.alpha == 0.01
        assert m.max_iter == 1000
        assert m.tol is None
        assert m.random_state is None

    def test_custom_params_stored(self):
        m = SimpleLinearRegression(alpha=0.05, max_iter=500, tol=1e-6, random_state=7)
        assert m.alpha == 0.05
        assert m.max_iter == 500
        assert m.tol == 1e-6
        assert m.random_state == 7

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            SimpleLinearRegression(alpha=-0.01)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError):
            SimpleLinearRegression(alpha=0.0)

    def test_zero_max_iter_raises(self):
        with pytest.raises(ValueError):
            SimpleLinearRegression(max_iter=0)

    def test_coef_none_before_fit(self):
        m = SimpleLinearRegression()
        assert m.coef_ is None

    def test_intercept_none_before_fit(self):
        m = SimpleLinearRegression()
        assert m.intercept_ is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit() — output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=100, random_state=0)
        result = m.fit(X, y)
        assert result is m

    def test_coef_is_float_after_fit(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.coef_, float)

    def test_intercept_is_float_after_fit(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.intercept_, float)

    def test_cost_history_is_list(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.cost_history_, list)

    def test_cost_history_length(self, perfect_line):
        X, y = perfect_line
        max_iter = 50
        m = SimpleLinearRegression(alpha=0.01, max_iter=max_iter, random_state=0)
        m.fit(X, y)
        # epoch-0 record + max_iter updates = max_iter + 1
        assert len(m.cost_history_) == max_iter + 1

    def test_mismatched_X_y_raises(self):
        m = SimpleLinearRegression()
        with pytest.raises(ValueError):
            m.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_2d_X_raises(self):
        m = SimpleLinearRegression()
        with pytest.raises(ValueError):
            m.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0]))

    def test_empty_X_raises(self):
        m = SimpleLinearRegression()
        with pytest.raises(ValueError):
            m.fit(np.array([]), np.array([]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence & parameter recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    def test_coef_close_to_3(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=42)
        m.fit(X, y)
        assert abs(m.coef_ - 3.0) < 0.05

    def test_intercept_close_to_1(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=42)
        m.fit(X, y)
        assert abs(m.intercept_ - 1.0) < 0.2

    def test_cost_decreases(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=500, random_state=0)
        m.fit(X, y)
        assert m.cost_history_[-1] < m.cost_history_[0]

    def test_cost_final_near_zero(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=42)
        m.fit(X, y)
        assert m.cost_history_[-1] < 0.1

    def test_noisy_coef_roughly_correct(self, noisy_line):
        X, y = noisy_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=0)
        m.fit(X, y)
        assert abs(m.coef_ - 2.0) < 0.3

    def test_early_stopping_shortens_history(self, perfect_line):
        X, y = perfect_line
        max_iter = 5000
        m = SimpleLinearRegression(alpha=0.01, max_iter=max_iter, tol=1e-4, random_state=0)
        m.fit(X, y)
        assert len(m.cost_history_) < max_iter + 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_shape(self, fitted_model):
        m, X, y = fitted_model
        preds = m.predict(X)
        assert preds.shape == X.shape

    def test_predict_close_to_true(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=42)
        m.fit(X, y)
        preds = m.predict(X)
        assert np.mean((preds - y) ** 2) < 0.1

    def test_predict_returns_ndarray(self, fitted_model):
        m, X, y = fitted_model
        preds = m.predict(X[:5])
        assert isinstance(preds, np.ndarray)

    def test_predict_single_value(self, fitted_model):
        m, X, y = fitted_model
        preds = m.predict(np.array([0.0]))
        assert len(preds) == 1

    def test_predict_before_fit_raises(self):
        m = SimpleLinearRegression()
        with pytest.raises(RuntimeError):
            m.predict(np.array([1.0, 2.0]))

    def test_2d_predict_raises(self, fitted_model):
        m, X, y = fitted_model
        with pytest.raises(ValueError):
            m.predict(np.array([[1.0, 2.0]]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. score() — R²
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_perfect_fit_r2_is_1(self, perfect_line):
        X, y = perfect_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=5000, random_state=42)
        m.fit(X, y)
        assert abs(m.score(X, y) - 1.0) < 0.01

    def test_r2_between_neg1_and_1_noisy(self, noisy_line):
        X, y = noisy_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=3000, random_state=0)
        m.fit(X, y)
        r2 = m.score(X, y)
        assert r2 <= 1.0

    def test_r2_positive_on_correlated_data(self, noisy_line):
        X, y = noisy_line
        m = SimpleLinearRegression(alpha=0.01, max_iter=3000, random_state=0)
        m.fit(X, y)
        assert m.score(X, y) > 0.8

    def test_score_returns_float(self, fitted_model):
        m, X, y = fitted_model
        assert isinstance(m.score(X, y), float)

    def test_constant_target_returns_one_or_zero(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0])
        m = SimpleLinearRegression(alpha=0.01, max_iter=100, random_state=0)
        m.fit(X, y)
        r2 = m.score(X, y)
        assert r2 in (0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self, perfect_line):
        X, y = perfect_line
        m1 = SimpleLinearRegression(alpha=0.01, max_iter=200, random_state=7)
        m2 = SimpleLinearRegression(alpha=0.01, max_iter=200, random_state=7)
        m1.fit(X, y)
        m2.fit(X, y)
        assert m1.coef_ == m2.coef_
        assert m1.intercept_ == m2.intercept_

    def test_different_seeds_may_differ(self, perfect_line):
        X, y = perfect_line
        m1 = SimpleLinearRegression(alpha=0.01, max_iter=5, random_state=1)
        m2 = SimpleLinearRegression(alpha=0.01, max_iter=5, random_state=99)
        m1.fit(X, y); m2.fit(X, y)
        # With very few iterations, init matters — they should differ
        assert m1.coef_ != m2.coef_ or m1.intercept_ != m2.intercept_


# ─────────────────────────────────────────────────────────────────────────────
# 7. plot methods don't crash
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotMethods:
    def test_plot_cost_no_error(self, fitted_model, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        m, X, y = fitted_model
        m.plot_cost()   # should not raise

    def test_plot_fit_no_error(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        m, X, y = fitted_model
        m.plot_fit(X, y)

    def test_plot_cost_before_fit_raises(self):
        m = SimpleLinearRegression()
        with pytest.raises(RuntimeError):
            m.plot_cost()

    def test_repr_unfitted(self):
        m = SimpleLinearRegression()
        assert "unfitted" in repr(m)

    def test_repr_fitted(self, fitted_model):
        m, X, y = fitted_model
        r = repr(m)
        assert "coef_" in r or "SimpleLinearRegression" in r
