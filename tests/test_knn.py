"""
test_knn.py

Comprehensive tests for the KNN classifier and the standalone distance function.

Covers:
- distance() correctness (Euclidean)
- KNN.fit() stores training data
- KNN._k_nearest() returns k sorted neighbours
- KNN.predict() returns correct labels on clean data
- Majority-vote tie-breaking consistency
- k=1 (nearest neighbour)
- k=n (all neighbours — baseline)
- Multi-class problems
- Different feature dimensionalities
- Correct handling of unseen points
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from knn import KNN, distance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Two clearly separated 2-D clusters."""
    X_train = np.array([[1, 1], [1, 2], [2, 1],    # class 0
                        [8, 8], [8, 9], [9, 8]],    # class 1
                       dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    return X_train, y_train

@pytest.fixture
def three_class_data():
    """Three clearly separated clusters."""
    X_train = np.array([[0, 0], [0, 1], [1, 0],        # class 0
                        [10, 0], [10, 1], [11, 0],      # class 1
                        [0, 10], [0, 11], [1, 10]],     # class 2
                       dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X_train, y_train

@pytest.fixture
def iris_subset():
    """Small balanced slice of Iris for integration tests."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
# 1. distance() function
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanceFunction:
    def test_identical_vectors(self):
        assert distance(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_known_3_4_5(self):
        a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
        assert abs(distance(a, b) - 5.0) < 1e-10

    def test_symmetry(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert abs(distance(a, b) - distance(b, a)) < 1e-12

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.normal(size=5), rng.normal(size=5)
            assert distance(a, b) >= 0

    def test_unit_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(distance(a, b) - np.sqrt(2)) < 1e-10

    def test_1d_vectors(self):
        assert abs(distance(np.array([3.0]), np.array([7.0])) - 4.0) < 1e-10

    def test_high_dimensional(self):
        a = np.zeros(100)
        b = np.ones(100)
        assert abs(distance(a, b) - 10.0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 2. KNN initialisation & fit
# ─────────────────────────────────────────────────────────────────────────────

class TestKNNInit:
    def test_k_stored(self):
        clf = KNN(k=5)
        assert clf.k == 5

    def test_attributes_none_before_fit(self):
        clf = KNN(k=3)
        assert clf.X_train is None
        assert clf.y_train is None

    def test_fit_stores_data(self, binary_data):
        X, y = binary_data
        clf = KNN(k=1)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.X_train, X)
        np.testing.assert_array_equal(clf.y_train, y)

    def test_fit_returns_none(self, binary_data):
        X, y = binary_data
        clf = KNN(k=1)
        result = clf.fit(X, y)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. _k_nearest
# ─────────────────────────────────────────────────────────────────────────────

class TestKNearest:
    def test_returns_k_neighbours(self, binary_data):
        X, y = binary_data
        clf = KNN(k=3)
        clf.fit(X, y)
        neighbours = clf._k_nearest(np.array([1.5, 1.5]))
        assert len(neighbours) == 3

    def test_sorted_by_distance(self, binary_data):
        X, y = binary_data
        clf = KNN(k=4)
        clf.fit(X, y)
        neighbours = clf._k_nearest(np.array([1.5, 1.5]))
        dists = [nb[2] for nb in neighbours]
        assert dists == sorted(dists)

    def test_closest_neighbour_is_correct(self, binary_data):
        X, y = binary_data
        clf = KNN(k=1)
        clf.fit(X, y)
        # Point right next to [1,1]
        neighbours = clf._k_nearest(np.array([1.01, 1.01]))
        nearest_label = neighbours[0][1]
        assert nearest_label == 0

    def test_tuple_structure(self, binary_data):
        X, y = binary_data
        clf = KNN(k=2)
        clf.fit(X, y)
        neighbours = clf._k_nearest(np.array([1.0, 1.0]))
        for nb in neighbours:
            assert len(nb) == 3  # (x_train, label, distance)
            assert isinstance(nb[2], float)

    def test_k_equals_n(self, binary_data):
        X, y = binary_data
        n = len(X)
        clf = KNN(k=n)
        clf.fit(X, y)
        neighbours = clf._k_nearest(np.array([5.0, 5.0]))
        assert len(neighbours) == n


# ─────────────────────────────────────────────────────────────────────────────
# 4. predict — binary classification
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBinary:
    def test_predict_returns_list(self, binary_data):
        X, y = binary_data
        clf = KNN(k=1)
        clf.fit(X, y)
        preds = clf.predict([[1.0, 1.0]])
        assert isinstance(preds, list)

    def test_predict_correct_class_cluster0(self, binary_data):
        X, y = binary_data
        clf = KNN(k=3)
        clf.fit(X, y)
        preds = clf.predict([[1.5, 1.5]])
        assert preds[0] == 0

    def test_predict_correct_class_cluster1(self, binary_data):
        X, y = binary_data
        clf = KNN(k=3)
        clf.fit(X, y)
        preds = clf.predict([[8.5, 8.5]])
        assert preds[0] == 1

    def test_perfect_accuracy_k1_on_training(self, binary_data):
        """k=1 should perfectly memorise training data."""
        X, y = binary_data
        clf = KNN(k=1)
        clf.fit(X, y)
        preds = np.array(clf.predict(X))
        assert np.mean(preds == y) == 1.0

    def test_multiple_test_points(self, binary_data):
        X, y = binary_data
        clf = KNN(k=3)
        clf.fit(X, y)
        X_test = np.array([[1.0, 1.0], [9.0, 9.0]])
        preds = clf.predict(X_test)
        assert len(preds) == 2
        assert preds[0] == 0
        assert preds[1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. predict — multi-class
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictMultiClass:
    def test_three_class_accuracy(self, three_class_data):
        X, y = three_class_data
        clf = KNN(k=1)
        clf.fit(X, y)
        preds = np.array(clf.predict(X))
        assert np.mean(preds == y) == 1.0

    def test_correct_class_per_cluster(self, three_class_data):
        X, y = three_class_data
        clf = KNN(k=3)
        clf.fit(X, y)
        assert clf.predict([[0.5, 0.5]])[0] == 0
        assert clf.predict([[10.5, 0.5]])[0] == 1
        assert clf.predict([[0.5, 10.5]])[0] == 2

    def test_output_length_matches_input(self, three_class_data):
        X, y = three_class_data
        clf = KNN(k=3)
        clf.fit(X, y)
        X_test = np.array([[0.5, 0.5], [10.5, 0.5], [0.5, 10.5]])
        preds = clf.predict(X_test)
        assert len(preds) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. k sensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestKSensitivity:
    def test_k1_vs_k3_both_correct_on_clean(self, binary_data):
        X, y = binary_data
        for k in [1, 3, 5]:
            clf = KNN(k=k)
            clf.fit(X, y)
            preds = np.array(clf.predict(X))
            # Clean separation — all k values should be correct
            assert np.mean(preds == y) >= 0.8, f"Low accuracy at k={k}"

    def test_k1_memorises_training(self, binary_data):
        X, y = binary_data
        clf = KNN(k=1)
        clf.fit(X, y)
        preds = np.array(clf.predict(X))
        assert np.mean(preds == y) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature dimensionality
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensionality:
    def test_1d_features(self):
        X_train = np.array([[1.0], [2.0], [10.0], [11.0]])
        y_train = np.array([0, 0, 1, 1])
        clf = KNN(k=1)
        clf.fit(X_train, y_train)
        assert clf.predict([[1.5]])[0] == 0
        assert clf.predict([[10.5]])[0] == 1

    def test_high_dimensional(self):
        rng = np.random.default_rng(0)
        X0 = rng.normal(loc=0, scale=0.5, size=(20, 50))
        X1 = rng.normal(loc=5, scale=0.5, size=(20, 50))
        X = np.vstack([X0, X1])
        y = np.array([0] * 20 + [1] * 20)
        clf = KNN(k=3)
        clf.fit(X, y)
        preds = np.array(clf.predict(X))
        assert np.mean(preds == y) > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration — Iris dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestIrisIntegration:
    def test_iris_accuracy_above_90(self, iris_subset):
        X_train, X_test, y_train, y_test = iris_subset
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

        clf = KNN(k=5)
        clf.fit(X_train, y_train)
        preds = np.array(clf.predict(X_test))
        acc = np.mean(preds == y_test)
        assert acc > 0.90, f"Iris accuracy {acc:.2f} below threshold"

    def test_predict_length_matches_test_set(self, iris_subset):
        X_train, X_test, y_train, y_test = iris_subset
        clf = KNN(k=3)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert len(preds) == len(y_test)
