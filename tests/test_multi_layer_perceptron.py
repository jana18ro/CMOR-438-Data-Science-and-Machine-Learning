"""
test_multi_layer_perceptron.py

Comprehensive tests for multi_layer_perceptron.

Covers:
- Constructor and parameter initialisation
- _initialize_parameters — weight/bias shapes
- Activation functions: sigmoid, sigmoid_derivative, softmax
- _one_hot_encode correctness
- _forward pass: activations list length and output probabilities
- _backward pass: gradient shapes
- fit() convergence and loss history
- predict_proba() — range, shape, sums to 1
- predict() — correct labels, shape
- score() accuracy
- Integration on MNIST digits (sklearn digits)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from multi_layer_perceptron import multi_layer_perceptron


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def xor_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y

@pytest.fixture
def binary_blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.5, (40, 2))
    X1 = rng.normal([5, 5], 0.5, (40, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0]*40 + [1]*40)
    return X, y

@pytest.fixture
def digits_split():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

@pytest.fixture
def mlp_small():
    return multi_layer_perceptron(
        layer_sizes=[2, 4, 2],
        learning_rate=0.1,
        epochs=1,
        batch_size=4,
        random_state=42
    )

@pytest.fixture
def fitted_binary(binary_blobs):
    X, y = binary_blobs
    m = multi_layer_perceptron(
        layer_sizes=[2, 8, 2],
        learning_rate=0.1,
        epochs=30,
        batch_size=16,
        random_state=42
    )
    m.fit(X, y)
    return m, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor & parameter initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_layer_sizes_stored(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        assert m.layer_sizes == [4, 8, 3]

    def test_learning_rate_stored(self):
        m = multi_layer_perceptron(layer_sizes=[2, 2], learning_rate=0.05)
        assert m.learning_rate == 0.05

    def test_epochs_stored(self):
        m = multi_layer_perceptron(layer_sizes=[2, 2], epochs=15)
        assert m.epochs == 15

    def test_batch_size_stored(self):
        m = multi_layer_perceptron(layer_sizes=[2, 2], batch_size=64)
        assert m.batch_size == 64

    def test_weights_length(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        assert len(m.weights) == 2   # one per layer pair

    def test_biases_length(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        assert len(m.biases) == 2

    def test_weight_shapes(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        assert m.weights[0].shape == (4, 8)
        assert m.weights[1].shape == (8, 3)

    def test_bias_shapes(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        assert m.biases[0].shape == (1, 8)
        assert m.biases[1].shape == (1, 3)

    def test_biases_initialised_to_zero(self):
        m = multi_layer_perceptron(layer_sizes=[4, 8, 3])
        for b in m.biases:
            np.testing.assert_array_equal(b, 0)

    def test_loss_history_empty_before_fit(self):
        m = multi_layer_perceptron(layer_sizes=[2, 2])
        assert m.loss_history == []

    def test_classes_none_before_fit(self):
        m = multi_layer_perceptron(layer_sizes=[2, 2])
        assert m.classes_ is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Activation functions
# ─────────────────────────────────────────────────────────────────────────────

class TestActivations:
    def _m(self):
        return multi_layer_perceptron(layer_sizes=[2, 2])

    def test_sigmoid_range(self):
        m = self._m()
        # Use moderate inputs that don't saturate in float64
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        s = m._sigmoid(z)
        assert np.all(s > 0) and np.all(s < 1)

    def test_sigmoid_at_zero(self):
        m = self._m()
        assert abs(m._sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        m = self._m()
        assert m._sigmoid(np.array([500.0]))[0] > 0.99

    def test_sigmoid_large_negative(self):
        m = self._m()
        assert m._sigmoid(np.array([-500.0]))[0] < 0.01

    def test_sigmoid_no_nan(self):
        m = self._m()
        z = np.array([-1000.0, 0.0, 1000.0])
        assert not np.any(np.isnan(m._sigmoid(z)))

    def test_sigmoid_derivative_range(self):
        m = self._m()
        a = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        d = m._sigmoid_derivative(a)
        assert np.all(d >= 0)

    def test_sigmoid_derivative_max_at_half(self):
        m = self._m()
        a = np.array([0.5])
        assert m._sigmoid_derivative(a)[0] == pytest.approx(0.25)

    def test_softmax_sums_to_one(self):
        m = self._m()
        z = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        s = m._softmax(z)
        np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-10)

    def test_softmax_nonnegative(self):
        m = self._m()
        z = np.random.randn(5, 4)
        assert np.all(m._softmax(z) >= 0)

    def test_softmax_no_nan_large_input(self):
        m = self._m()
        z = np.array([[1000.0, 0.0, -1000.0]])
        s = m._softmax(z)
        assert not np.any(np.isnan(s))


# ─────────────────────────────────────────────────────────────────────────────
# 3. One-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestOneHotEncode:
    def test_shape(self):
        m = multi_layer_perceptron(layer_sizes=[2, 3])
        y = np.array([0, 1, 2, 0, 1])
        ohe = m._one_hot_encode(y)
        assert ohe.shape == (5, 3)

    def test_correct_encoding(self):
        m = multi_layer_perceptron(layer_sizes=[2, 3])
        y = np.array([0, 1, 2])
        ohe = m._one_hot_encode(y)
        np.testing.assert_array_equal(ohe[0], [1, 0, 0])
        np.testing.assert_array_equal(ohe[1], [0, 1, 0])
        np.testing.assert_array_equal(ohe[2], [0, 0, 1])

    def test_row_sums_to_one(self):
        m = multi_layer_perceptron(layer_sizes=[2, 3])
        y = np.array([0, 1, 2, 1, 0])
        ohe = m._one_hot_encode(y)
        np.testing.assert_array_equal(ohe.sum(axis=1), 1)

    def test_classes_set(self):
        m = multi_layer_perceptron(layer_sizes=[2, 3])
        m._one_hot_encode(np.array([0, 1, 2]))
        np.testing.assert_array_equal(m.classes_, [0, 1, 2])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestForward:
    def test_activations_length(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        _, activations = m._forward(X)
        # n_layers + 1 (input + each layer)
        assert len(activations) == 3

    def test_output_probabilities_sum_to_one(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        _, activations = m._forward(X)
        output = activations[-1]
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-10)

    def test_output_shape(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        _, activations = m._forward(X)
        assert activations[-1].shape == (len(X), 2)

    def test_hidden_activations_in_0_1(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        _, activations = m._forward(X)
        for a in activations[1:-1]:
            assert np.all(a >= 0) and np.all(a <= 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestBackward:
    def test_gradient_shapes_match_weights(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        y_ohe = m._one_hot_encode(y)
        _, activations = m._forward(X)
        gw, gb = m._backward(y_ohe, activations)
        for gw_l, w_l in zip(gw, m.weights):
            assert gw_l.shape == w_l.shape
        for gb_l, b_l in zip(gb, m.biases):
            assert gb_l.shape == b_l.shape

    def test_gradients_not_nan(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], random_state=0)
        y_ohe = m._one_hot_encode(y)
        _, activations = m._forward(X)
        gw, gb = m._backward(y_ohe, activations)
        for gw_l in gw:
            assert not np.any(np.isnan(gw_l))


# ─────────────────────────────────────────────────────────────────────────────
# 6. fit()
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 4, 2], epochs=2, random_state=0)
        assert m.fit(X, y) is m

    def test_loss_history_length(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 4, 2], epochs=5, random_state=0)
        m.fit(X, y)
        assert len(m.loss_history) == 5

    def test_loss_decreases(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], learning_rate=0.1,
                                    epochs=50, batch_size=16, random_state=0)
        m.fit(X, y)
        assert m.loss_history[-1] < m.loss_history[0]

    def test_classes_set_after_fit(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 4, 2], epochs=2, random_state=0)
        m.fit(X, y)
        np.testing.assert_array_equal(m.classes_, [0, 1])

    def test_wrong_input_dims_raises(self):
        m = multi_layer_perceptron(layer_sizes=[2, 4, 2], epochs=1)
        with pytest.raises(ValueError):
            m.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_mismatched_X_y_raises(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 4, 2], epochs=1)
        with pytest.raises(ValueError):
            m.fit(X, y[:-1])


# ─────────────────────────────────────────────────────────────────────────────
# 7. predict_proba()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictProba:
    def test_shape(self, fitted_binary):
        m, X, y = fitted_binary
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_sums_to_one(self, fitted_binary):
        m, X, y = fitted_binary
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self, fitted_binary):
        m, X, y = fitted_binary
        assert np.all(m.predict_proba(X) >= 0)

    def test_wrong_input_dims_raises(self, fitted_binary):
        m, X, y = fitted_binary
        with pytest.raises(ValueError):
            m.predict_proba(np.array([1.0, 2.0]))


# ─────────────────────────────────────────────────────────────────────────────
# 8. predict() and score()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictAndScore:
    def test_predict_shape(self, fitted_binary):
        m, X, y = fitted_binary
        assert m.predict(X).shape == (len(X),)

    def test_predict_labels_in_classes(self, fitted_binary):
        m, X, y = fitted_binary
        preds = m.predict(X)
        assert all(p in m.classes_ for p in preds)

    def test_predict_high_accuracy_on_blobs(self, binary_blobs):
        X, y = binary_blobs
        m = multi_layer_perceptron(layer_sizes=[2, 16, 2], learning_rate=0.1,
                                    epochs=80, batch_size=16, random_state=42)
        m.fit(X, y)
        assert m.score(X, y) > 0.95

    def test_score_between_0_and_1(self, fitted_binary):
        m, X, y = fitted_binary
        assert 0 <= m.score(X, y) <= 1

    def test_score_equals_manual_accuracy(self, fitted_binary):
        m, X, y = fitted_binary
        preds = m.predict(X)
        assert abs(m.score(X, y) - np.mean(preds == y)) < 1e-10

    def test_xor_solvable_with_hidden_layer(self, xor_data):
        X, y = xor_data
        m = multi_layer_perceptron(layer_sizes=[2, 8, 2], learning_rate=0.5,
                                    epochs=500, batch_size=4, random_state=42)
        m.fit(X, y)
        assert m.score(X, y) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration — sklearn digits
# ─────────────────────────────────────────────────────────────────────────────

class TestDigitsIntegration:
    def test_accuracy_above_90(self, digits_split):
        X_train, X_test, y_train, y_test = digits_split
        m = multi_layer_perceptron(
            layer_sizes=[64, 64, 10],
            learning_rate=0.05,
            epochs=25,
            batch_size=32,
            random_state=42
        )
        m.fit(X_train, y_train)
        assert m.score(X_test, y_test) > 0.90

    def test_predict_length_matches_test(self, digits_split):
        X_train, X_test, y_train, y_test = digits_split
        m = multi_layer_perceptron(
            layer_sizes=[64, 32, 10],
            learning_rate=0.05,
            epochs=5,
            batch_size=32,
            random_state=0
        )
        m.fit(X_train, y_train)
        assert len(m.predict(X_test)) == len(y_test)
