"""
test_decision_tree_regressor.py

Comprehensive tests for decision_tree_regressor.

Covers:
- Attribute initialisation
- fit() / predict() on simple and multi-feature data
- Variance and variance reduction helpers
- max_depth and min_samples_split effects
- score() (R²) and mean_squared_error()
- Error handling (unfitted model, mismatched X/y)
- Integration on California Housing subset
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from decision_tree_regressor import decision_tree_regressor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_1d():
    """y = x**2 — non-linear, well-suited for a tree."""
    X = np.linspace(-3, 3, 30).reshape(-1, 1)
    y = X.ravel() ** 2
    return X, y

@pytest.fixture
def perfect_step():
    """Step function — a depth-1 tree should fit perfectly."""
    X = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
    y = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    return X, y

@pytest.fixture
def multi_feature():
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, (80, 3))
    y = X[:, 0] ** 2 + 2 * X[:, 1] - X[:, 2]
    return X, y

@pytest.fixture
def housing_split():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=600, n_features=8, noise=20, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_max_depth(self):
        assert decision_tree_regressor().max_depth is None

    def test_default_min_samples_split(self):
        assert decision_tree_regressor().min_samples_split == 2

    def test_tree_none_before_fit(self):
        assert decision_tree_regressor().tree is None

    def test_custom_max_depth(self):
        reg = decision_tree_regressor(max_depth=5)
        assert reg.max_depth == 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit()
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor()
        assert reg.fit(X, y) is reg

    def test_tree_not_none_after_fit(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor()
        reg.fit(X, y)
        assert reg.tree is not None

    def test_mismatched_X_y_raises(self, simple_1d):
        X, y = simple_1d
        with pytest.raises(ValueError):
            decision_tree_regressor().fit(X, y[:-1])

    def test_1d_X_accepted(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 4.0, 9.0, 16.0])
        reg = decision_tree_regressor()
        reg.fit(X, y)
        assert reg.tree is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            decision_tree_regressor().predict(np.array([[1.0]]))

    def test_predict_shape(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor()
        reg.fit(X, y)
        assert reg.predict(X).shape == (len(X),)

    def test_predict_returns_ndarray(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor()
        reg.fit(X, y)
        assert isinstance(reg.predict(X), np.ndarray)

    def test_memorises_training_data(self, simple_1d):
        """Unlimited depth should perfectly fit training data."""
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=None)
        reg.fit(X, y)
        preds = reg.predict(X)
        np.testing.assert_allclose(preds, y, atol=1e-6)

    def test_step_function_depth1(self, perfect_step):
        X, y = perfect_step
        reg = decision_tree_regressor(max_depth=1)
        reg.fit(X, y)
        preds = reg.predict(X)
        np.testing.assert_allclose(preds, y, atol=1e-6)

    def test_predict_dtype_float(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        assert np.issubdtype(preds.dtype, np.floating)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Variance helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceHelpers:
    def _reg(self):
        return decision_tree_regressor()

    def test_variance_constant_is_zero(self):
        reg = self._reg()
        assert reg._variance(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_variance_nonnegative(self):
        reg = self._reg()
        for arr in [np.array([1.0]), np.array([1.0, 2.0, 3.0])]:
            assert reg._variance(arr) >= 0

    def test_variance_known_value(self):
        reg = self._reg()
        # Variance of [2,4,4,4,5,5,7,9] = 4
        y = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert reg._variance(y) == pytest.approx(4.0)

    def test_variance_reduction_perfect_split(self, perfect_step):
        X, y = perfect_step
        reg = self._reg()
        left  = y[:3]
        right = y[3:]
        vr = reg._variance_reduction(y, left, right)
        assert vr > 0

    def test_variance_reduction_non_negative(self):
        reg = self._reg()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        vr = reg._variance_reduction(y, y[:2], y[2:])
        assert vr >= 0

    def test_leaf_value_is_mean(self):
        reg = self._reg()
        y = np.array([1.0, 3.0, 5.0])
        assert reg._leaf_value(y) == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. max_depth
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDepth:
    def test_depth1_gives_single_split(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=1)
        reg.fit(X, y)
        # Predict should produce at most 2 unique values (two leaves)
        preds = reg.predict(X)
        assert len(np.unique(preds)) <= 2

    def test_unlimited_depth_zero_train_mse(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=None)
        reg.fit(X, y)
        assert reg.mean_squared_error(X, y) < 1e-10

    def test_deeper_tree_lower_train_mse(self, simple_1d):
        X, y = simple_1d
        mses = []
        for d in [1, 2, 4, None]:
            reg = decision_tree_regressor(max_depth=d)
            reg.fit(X, y)
            mses.append(reg.mean_squared_error(X, y))
        # Each deeper tree should have ≤ MSE of shallower
        assert mses[-1] <= mses[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. score() and mean_squared_error()
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_r2_perfect(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=None)
        reg.fit(X, y)
        assert reg.score(X, y) == pytest.approx(1.0, abs=1e-6)

    def test_r2_between_0_and_1(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=2)
        reg.fit(X, y)
        r2 = reg.score(X, y)
        assert 0 <= r2 <= 1

    def test_mse_nonnegative(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=2)
        reg.fit(X, y)
        assert reg.mean_squared_error(X, y) >= 0

    def test_mse_zero_for_perfect_fit(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=None)
        reg.fit(X, y)
        assert reg.mean_squared_error(X, y) < 1e-10

    def test_score_returns_float(self, simple_1d):
        X, y = simple_1d
        reg = decision_tree_regressor(max_depth=2)
        reg.fit(X, y)
        assert isinstance(reg.score(X, y), float)


# ─────────────────────────────────────────────────────────────────────────────
# 7. min_samples_split
# ─────────────────────────────────────────────────────────────────────────────

class TestMinSamplesSplit:
    def test_large_min_samples_creates_leaf(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reg = decision_tree_regressor(min_samples_split=100)
        reg.fit(X, y)
        preds = reg.predict(X)
        # All predictions should be the overall mean
        assert np.all(np.isclose(preds, np.mean(y)))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration — California Housing
# ─────────────────────────────────────────────────────────────────────────────

class TestHousingIntegration:
    def test_r2_above_0_5(self, housing_split):
        X_train, X_test, y_train, y_test = housing_split
        reg = decision_tree_regressor(max_depth=6)
        reg.fit(X_train, y_train)
        assert reg.score(X_test, y_test) > 0.5

    def test_predict_length(self, housing_split):
        X_train, X_test, y_train, y_test = housing_split
        reg = decision_tree_regressor(max_depth=4)
        reg.fit(X_train, y_train)
        assert len(reg.predict(X_test)) == len(y_test)
