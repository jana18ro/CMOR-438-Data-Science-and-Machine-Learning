"""
test_gradient_descent.py

Comprehensive tests for the GradientDescent optimiser.

Covers:
- Correct parameter initialisation
- Cost history structure (length, monotone decrease on convex problems)
- Convergence on simple MSE (linear regression) problems
- Early stopping via tol
- Reproducibility via random_state
- inject-ability of custom cost / gradient functions
- Edge cases: single feature, single sample, tol=0
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/Jana CMOR/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from rice_ml.supervised_learning.gradient_descent import GradientDescent

# ─────────────────────────────────────────────────────────────────────────────
# Minimal MSE cost / gradient helpers (mirror linear_regression internals)
# ─────────────────────────────────────────────────────────────────────────────

def mse_cost(X, y, w, b):
    residuals = y - (X @ w + b)
    return float(np.mean(residuals ** 2))

def mse_gradients(X, y, w, b):
    n = len(y)
    error = y - (X @ w + b)
    dw = (-2 / n) * (X.T @ error)
    db = float((-2 / n) * np.sum(error))
    return dw, db

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_linear_data():
    """y = 3x + 2 — noiseless, easy to converge."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (60, 1))
    y = 3 * X.ravel() + 2
    return X, y

@pytest.fixture
def multi_feature_data():
    """y = 2x1 + 5x2 + 1, shape (80, 2)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (80, 2))
    y = 2 * X[:, 0] + 5 * X[:, 1] + 1
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_attributes(self):
        gd = GradientDescent()
        assert gd.alpha == 0.01
        assert gd.max_iter == 1000
        assert gd.tol is None
        assert gd.random_state is None

    def test_custom_params(self):
        gd = GradientDescent(alpha=0.5, max_iter=200, tol=1e-6, random_state=7)
        assert gd.alpha == 0.5
        assert gd.max_iter == 200
        assert gd.tol == 1e-6
        assert gd.random_state == 7

    def test_weights_none_before_optimize(self):
        gd = GradientDescent()
        assert gd.weights_ is None
        assert gd.bias_ is None
        assert gd.cost_history_ == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Return types & shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnTypes:
    def test_returns_tuple_of_three(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        result = gd.optimize(X, y, mse_cost, mse_gradients)
        assert isinstance(result, tuple) and len(result) == 3

    def test_weights_shape(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert w.shape == (1,)

    def test_bias_is_float(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        _, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert isinstance(b, float)

    def test_cost_history_is_list(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=30, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert isinstance(hist, list)

    def test_cost_history_length_no_early_stop(self, simple_linear_data):
        """History = epoch-0 record + max_iter records = max_iter + 1."""
        X, y = simple_linear_data
        max_iter = 40
        gd = GradientDescent(alpha=0.05, max_iter=max_iter, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert len(hist) == max_iter + 1

    def test_weights_shape_multi_feature(self, multi_feature_data):
        X, y = multi_feature_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        w, _, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert w.shape == (2,)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cost history behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCostHistory:
    def test_cost_history_all_nonnegative(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=100, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert all(c >= 0 for c in hist)

    def test_cost_decreases_overall(self, simple_linear_data):
        """Final cost should be much smaller than initial cost."""
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=500, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert hist[-1] < hist[0]

    def test_cost_history_stored_on_instance(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert gd.cost_history_ is hist

    def test_initial_cost_recorded_before_any_update(self, simple_linear_data):
        """First element of cost_history_ is cost at initial weights."""
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=10, random_state=42)
        gd.optimize(X, y, mse_cost, mse_gradients)
        # Reset and check initial cost matches manually
        np.random.seed(42)
        w0 = np.random.randn(1) * 0.01
        b0 = float(np.random.randn() * 0.01)
        expected_initial = mse_cost(X, y, w0, b0)
        assert abs(gd.cost_history_[0] - expected_initial) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. Convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    def test_converges_on_noiseless_linear(self, simple_linear_data):
        """After enough iterations the MSE should be near 0."""
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=3000, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        final_mse = mse_cost(X, y, w, b)
        assert final_mse < 0.01

    def test_learned_slope_close_to_3(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=3000, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert abs(float(w[0]) - 3.0) < 0.1

    def test_learned_intercept_close_to_2(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=3000, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert abs(b - 2.0) < 0.2

    def test_converges_multi_feature(self, multi_feature_data):
        X, y = multi_feature_data
        gd = GradientDescent(alpha=0.05, max_iter=5000, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        final_mse = mse_cost(X, y, w, b)
        assert final_mse < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 5. Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_early_stop_shorter_history(self, simple_linear_data):
        """With tight tol, training should stop before max_iter."""
        X, y = simple_linear_data
        max_iter = 5000
        gd = GradientDescent(alpha=0.05, max_iter=max_iter, tol=1e-3, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert len(hist) < max_iter + 1

    def test_no_early_stop_without_tol(self, simple_linear_data):
        X, y = simple_linear_data
        max_iter = 100
        gd = GradientDescent(alpha=0.05, max_iter=max_iter, tol=None, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert len(hist) == max_iter + 1

    def test_early_stop_still_converges(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=5000, tol=1e-8, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert mse_cost(X, y, w, b) < 0.01

    def test_weights_stored_after_early_stop(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=5000, tol=1e-3, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        assert gd.weights_ is not None
        assert gd.bias_ is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self, simple_linear_data):
        X, y = simple_linear_data
        gd1 = GradientDescent(alpha=0.05, max_iter=100, random_state=42)
        gd2 = GradientDescent(alpha=0.05, max_iter=100, random_state=42)
        w1, b1, h1 = gd1.optimize(X, y, mse_cost, mse_gradients)
        w2, b2, h2 = gd2.optimize(X, y, mse_cost, mse_gradients)
        np.testing.assert_array_equal(w1, w2)
        assert b1 == b2
        assert h1 == h2

    def test_different_seeds_different_init(self, simple_linear_data):
        X, y = simple_linear_data
        gd1 = GradientDescent(alpha=0.0, max_iter=0, random_state=1)
        gd2 = GradientDescent(alpha=0.0, max_iter=0, random_state=99)
        w1, b1, _ = gd1.optimize(X, y, mse_cost, mse_gradients)
        w2, b2, _ = gd2.optimize(X, y, mse_cost, mse_gradients)
        # With 0 iterations weights stay at init — different seeds → different inits
        assert not (np.allclose(w1, w2) and b1 == b2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Custom cost / gradient injection
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomFunctions:
    def test_custom_quadratic_converges(self):
        """Minimise f(w) = (w - 5)^2. Gradient = 2*(w-5). Minimum at w=5."""
        X = np.ones((10, 1))
        y = np.full(10, 5.0)

        def quad_cost(X, y, w, b):
            return float(np.mean((w[0] - 5.0) ** 2))

        def quad_grad(X, y, w, b):
            return np.array([2 * (w[0] - 5.0)]), 0.0

        gd = GradientDescent(alpha=0.1, max_iter=500, random_state=0)
        w, _, _ = gd.optimize(X, y, quad_cost, quad_grad)
        assert abs(float(w[0]) - 5.0) < 0.1

    def test_cost_function_called_each_iteration(self, simple_linear_data):
        """Verify the cost function is called at every iteration."""
        X, y = simple_linear_data
        call_count = [0]

        def counting_cost(X, y, w, b):
            call_count[0] += 1
            return mse_cost(X, y, w, b)

        max_iter = 20
        gd = GradientDescent(alpha=0.05, max_iter=max_iter, random_state=0)
        gd.optimize(X, y, counting_cost, mse_gradients)
        # Called once before loop + once per iteration
        assert call_count[0] == max_iter + 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_iterations(self, simple_linear_data):
        """max_iter=0 → only initial cost recorded, weights stay at init."""
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=0, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert len(hist) == 1

    def test_single_sample(self):
        X = np.array([[2.0]])
        y = np.array([6.0])
        gd = GradientDescent(alpha=0.1, max_iter=1000, random_state=0)
        w, b, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert hist[-1] < hist[0]

    def test_weights_updated_on_instance(self, simple_linear_data):
        X, y = simple_linear_data
        gd = GradientDescent(alpha=0.05, max_iter=50, random_state=0)
        w, b, _ = gd.optimize(X, y, mse_cost, mse_gradients)
        np.testing.assert_array_equal(gd.weights_, w)
        assert gd.bias_ == b

    def test_high_learning_rate_diverges_gracefully(self, simple_linear_data):
        """Very large alpha diverges; cost history still has correct length."""
        X, y = simple_linear_data
        gd = GradientDescent(alpha=100.0, max_iter=10, random_state=0)
        _, _, hist = gd.optimize(X, y, mse_cost, mse_gradients)
        assert len(hist) == 11
