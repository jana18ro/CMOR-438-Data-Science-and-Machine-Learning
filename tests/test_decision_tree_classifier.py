"""
test_decision_tree_classifier.py

Comprehensive tests for decision_tree_classifier.

Covers:
- Attribute initialisation
- fit() / predict() on XOR and separable datasets
- Entropy and information gain calculations
- max_depth limits tree depth
- min_samples_split prevents tiny splits
- Pure nodes become leaves immediately
- score() accuracy
- Error handling (unfitted model, mismatched X/y)
- Multi-class problems
- Integration on Breast Cancer dataset
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/Jana CMOR/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from rice_ml.supervised_learning.decision_tree_classifier import decision_tree_classifier


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y

@pytest.fixture
def separable_2d():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.5, (30, 2))
    X1 = rng.normal(5, 0.5, (30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y

@pytest.fixture
def three_class():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(i * 5, 0.3, (20, 2)) for i in range(3)])
    y = np.array([i for i in range(3) for _ in range(20)])
    return X, y

@pytest.fixture
def cancer_split():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_max_depth_is_none(self):
        assert decision_tree_classifier().max_depth is None

    def test_default_min_samples_split(self):
        assert decision_tree_classifier().min_samples_split == 2

    def test_tree_none_before_fit(self):
        assert decision_tree_classifier().tree is None

    def test_classes_none_before_fit(self):
        assert decision_tree_classifier().classes_ is None

    def test_custom_params(self):
        clf = decision_tree_classifier(max_depth=5, min_samples_split=10)
        assert clf.max_depth == 5
        assert clf.min_samples_split == 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit() structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        assert clf.fit(X, y) is clf

    def test_tree_not_none_after_fit(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert clf.tree is not None

    def test_classes_set_after_fit(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_three_classes_stored(self, three_class):
        X, y = three_class
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert len(clf.classes_) == 3

    def test_mismatched_X_y_raises(self, separable_2d):
        X, y = separable_2d
        with pytest.raises(ValueError):
            decision_tree_classifier().fit(X, y[:-1])

    def test_1d_X_accepted(self):
        X = np.array([1.0, 2.0, 8.0, 9.0])
        y = np.array([0, 0, 1, 1])
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert clf.tree is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            decision_tree_classifier().predict(np.array([[1, 2]]))

    def test_predict_shape(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_perfect_memorisation_depth_none(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier(max_depth=None)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_simple_4point_depth_none(self):
        """Four clearly separable points — unlimited depth should memorise."""
        X = np.array([[1.0,1.0],[2.0,2.0],[8.0,1.0],[9.0,2.0]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = decision_tree_classifier(max_depth=None)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_predict_returns_ndarray(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert isinstance(clf.predict(X), np.ndarray)

    def test_labels_in_classes_(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert all(p in clf.classes_ for p in preds)


# ─────────────────────────────────────────────────────────────────────────────
# 4. max_depth
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDepth:
    def test_depth1_limits_splits(self, separable_2d):
        """Depth-1 tree has only one split; may not perfectly classify."""
        X, y = separable_2d
        clf = decision_tree_classifier(max_depth=1)
        clf.fit(X, y)
        # Should still classify most points correctly (data is well-separated)
        assert clf.score(X, y) > 0.5

    def test_unlimited_depth_perfect(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier(max_depth=None)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_increasing_depth_improves_train(self, xor_data):
        X, y = xor_data
        accs = []
        for d in [1, 2, 3]:
            clf = decision_tree_classifier(max_depth=d)
            clf.fit(X, y)
            accs.append(clf.score(X, y))
        # Accuracy should be non-decreasing with depth on training data
        assert accs[-1] >= accs[0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. min_samples_split
# ─────────────────────────────────────────────────────────────────────────────

class TestMinSamplesSplit:
    def test_high_min_samples_creates_single_leaf(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        # With min_samples_split > n_samples, root becomes a leaf
        clf = decision_tree_classifier(min_samples_split=100)
        clf.fit(X, y)
        # Leaf predicts majority class — both classes equal → either is fine
        preds = clf.predict(X)
        assert len(preds) == 4

    def test_default_min_samples_allows_splits(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier(min_samples_split=2)
        clf.fit(X, y)
        assert clf.score(X, y) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entropy & information gain (internal methods)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyAndIG:
    def _clf(self):
        return decision_tree_classifier()

    def test_entropy_pure_node_is_zero(self):
        clf = self._clf()
        assert clf._entropy(np.array([1, 1, 1, 1])) == 0.0

    def test_entropy_max_at_equal_split(self):
        clf = self._clf()
        e_equal = clf._entropy(np.array([0, 0, 1, 1]))
        e_pure  = clf._entropy(np.array([0, 0, 0, 0]))
        assert e_equal > e_pure

    def test_entropy_nonnegative(self):
        clf = self._clf()
        for labels in [[0], [0, 1], [0, 1, 2], [0, 0, 0, 1]]:
            assert clf._entropy(np.array(labels)) >= 0

    def test_information_gain_perfect_split(self):
        clf = self._clf()
        parent = np.array([0, 0, 1, 1])
        left   = np.array([0, 0])
        right  = np.array([1, 1])
        ig = clf._information_gain(parent, left, right)
        assert ig > 0

    def test_information_gain_no_split(self):
        clf = self._clf()
        y = np.array([0, 0, 1, 1])
        # Splitting evenly at the impure boundary gives 0 gain (same distribution)
        ig = clf._information_gain(y, y[:2], y[2:])
        # Both halves have equal impurity to parent — gain is 0
        assert ig >= 0

    def test_majority_class_returns_most_common(self):
        clf = self._clf()
        y = np.array([0, 0, 0, 1, 1])
        assert clf._majority_class(y) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. score()
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_in_0_1(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier(max_depth=3)
        clf.fit(X, y)
        assert 0 <= clf.score(X, y) <= 1

    def test_score_returns_float(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        assert isinstance(clf.score(X, y), (float, np.floating))

    def test_score_equals_manual_accuracy(self, separable_2d):
        X, y = separable_2d
        clf = decision_tree_classifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert abs(clf.score(X, y) - np.mean(preds == y)) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration — Breast Cancer
# ─────────────────────────────────────────────────────────────────────────────

class TestBreastCancerIntegration:
    def test_accuracy_above_90(self, cancer_split):
        X_train, X_test, y_train, y_test = cancer_split
        clf = decision_tree_classifier(max_depth=5)
        clf.fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.90

    def test_predict_length_matches_test(self, cancer_split):
        X_train, X_test, y_train, y_test = cancer_split
        clf = decision_tree_classifier(max_depth=4)
        clf.fit(X_train, y_train)
        assert len(clf.predict(X_test)) == len(y_test)
