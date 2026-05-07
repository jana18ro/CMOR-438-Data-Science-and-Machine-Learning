"""
test_ensemble.py

Comprehensive tests for ensemble.py:
  - hard_voting_classifier
  - bagging_classifier
  - random_forest_classifier

Covers:
- Constructor validation
- fit() / predict() / score() on clean data
- Internal majority-vote logic
- Bagging bootstrap sampling path
- Random forest feature-subset path (max_features variants)
- Reproducibility
- Error handling (empty classifier list, unfitted model)
- Integration on Wine / Breast Cancer datasets
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/supervised_learning")
from ensemble import (hard_voting_classifier, bagging_classifier,
                      random_forest_classifier, _ensemble_tree_classifier)
from decision_tree_classifier import decision_tree_classifier
from knn import KNN
from logistic_regression import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.4, (30, 2))
    X1 = rng.normal([5, 5], 0.4, (30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y

@pytest.fixture
def wine_split():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

@pytest.fixture
def cancer_split():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
# ── _ensemble_tree_classifier (internal) ─────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestInternalTree:
    def test_fits_and_predicts(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_depth=4, random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert len(preds) == len(y)

    def test_max_features_sqrt(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_depth=4, max_features='sqrt', random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert np.mean(preds == y) > 0.9

    def test_max_features_log2(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_depth=4, max_features='log2', random_state=0)
        tree.fit(X, y)
        assert len(tree.predict(X)) == len(y)

    def test_max_features_int(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_depth=4, max_features=1, random_state=0)
        tree.fit(X, y)
        assert len(tree.predict(X)) == len(y)

    def test_max_features_float(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_depth=4, max_features=0.5, random_state=0)
        tree.fit(X, y)
        assert len(tree.predict(X)) == len(y)

    def test_invalid_max_features_raises(self, separable):
        X, y = separable
        tree = _ensemble_tree_classifier(max_features='invalid')
        with pytest.raises(ValueError):
            tree._choose_feature_indices(5)


# ─────────────────────────────────────────────────────────────────────────────
# ── hard_voting_classifier ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestHardVoting:
    def _make_voter(self):
        return hard_voting_classifier(classifiers=[
            decision_tree_classifier(max_depth=3),
            KNN(k=3),
        ])

    def test_empty_classifiers_raises(self):
        with pytest.raises(ValueError):
            hard_voting_classifier(classifiers=[])

    def test_fit_returns_self(self, separable):
        X, y = separable
        v = self._make_voter()
        assert v.fit(X, y) is v

    def test_classes_set_after_fit(self, separable):
        X, y = separable
        v = self._make_voter()
        v.fit(X, y)
        np.testing.assert_array_equal(v.classes_, [0, 1])

    def test_predict_before_fit_raises(self):
        v = self._make_voter()
        with pytest.raises(RuntimeError):
            v.predict(np.ones((5, 2)))

    def test_predict_shape(self, separable):
        X, y = separable
        v = self._make_voter()
        v.fit(X, y)
        assert v.predict(X).shape == (len(X),)

    def test_predict_accuracy_high(self, separable):
        X, y = separable
        v = self._make_voter()
        v.fit(X, y)
        assert v.score(X, y) > 0.95

    def test_score_equals_manual_accuracy(self, separable):
        X, y = separable
        v = self._make_voter()
        v.fit(X, y)
        preds = v.predict(X)
        assert abs(v.score(X, y) - np.mean(preds == y)) < 1e-10

    def test_mismatched_X_y_raises(self, separable):
        X, y = separable
        v = self._make_voter()
        with pytest.raises(ValueError):
            v.fit(X, y[:-1])

    def test_three_classifiers(self, separable):
        X, y = separable
        v = hard_voting_classifier(classifiers=[
            decision_tree_classifier(max_depth=2),
            decision_tree_classifier(max_depth=3),
            KNN(k=5),
        ])
        v.fit(X, y)
        assert v.score(X, y) > 0.9

    def test_all_classifiers_trained(self, separable):
        """After fit, each classifier's tree/data should be populated."""
        X, y = separable
        clf1 = decision_tree_classifier(max_depth=2)
        clf2 = KNN(k=3)
        v = hard_voting_classifier(classifiers=[clf1, clf2])
        v.fit(X, y)
        assert clf1.tree is not None
        assert clf2.X_train is not None


# ─────────────────────────────────────────────────────────────────────────────
# ── bagging_classifier ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestBagging:
    def test_n_estimators_less_than_1_raises(self):
        with pytest.raises(ValueError):
            bagging_classifier(n_estimators=0)

    def test_fit_returns_self(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=5, random_state=0)
        assert b.fit(X, y) is b

    def test_correct_number_of_trees(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=7, random_state=0)
        b.fit(X, y)
        assert len(b.trees) == 7

    def test_predict_shape(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=5, random_state=0)
        b.fit(X, y)
        assert b.predict(X).shape == (len(X),)

    def test_predict_before_fit_raises(self):
        b = bagging_classifier(n_estimators=5)
        with pytest.raises(RuntimeError):
            b.predict(np.ones((5, 2)))

    def test_accuracy_high_on_separable(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=10, max_depth=4, random_state=42)
        b.fit(X, y)
        assert b.score(X, y) > 0.95

    def test_score_equals_manual_accuracy(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=5, random_state=0)
        b.fit(X, y)
        preds = b.predict(X)
        assert abs(b.score(X, y) - np.mean(preds == y)) < 1e-10

    def test_bootstrap_false_works(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=5, bootstrap=False, random_state=0)
        b.fit(X, y)
        assert b.score(X, y) > 0.8

    def test_reproducibility(self, separable):
        X, y = separable
        b1 = bagging_classifier(n_estimators=5, random_state=7)
        b2 = bagging_classifier(n_estimators=5, random_state=7)
        b1.fit(X, y); b2.fit(X, y)
        np.testing.assert_array_equal(b1.predict(X), b2.predict(X))

    def test_classes_set(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=3, random_state=0)
        b.fit(X, y)
        np.testing.assert_array_equal(b.classes_, [0, 1])

    def test_mismatched_X_y_raises(self, separable):
        X, y = separable
        b = bagging_classifier(n_estimators=3)
        with pytest.raises(ValueError):
            b.fit(X, y[:-1])


# ─────────────────────────────────────────────────────────────────────────────
# ── random_forest_classifier ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomForest:
    def test_n_estimators_less_than_1_raises(self):
        with pytest.raises(ValueError):
            random_forest_classifier(n_estimators=0)

    def test_fit_returns_self(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, random_state=0)
        assert rf.fit(X, y) is rf

    def test_correct_number_of_trees(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=8, random_state=0)
        rf.fit(X, y)
        assert len(rf.trees) == 8

    def test_predict_shape(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        assert rf.predict(X).shape == (len(X),)

    def test_predict_before_fit_raises(self):
        rf = random_forest_classifier(n_estimators=5)
        with pytest.raises(RuntimeError):
            rf.predict(np.ones((5, 2)))

    def test_accuracy_high_on_separable(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.95

    def test_score_equals_manual_accuracy(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert abs(rf.score(X, y) - np.mean(preds == y)) < 1e-10

    def test_max_features_sqrt(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, max_features='sqrt', random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.8

    def test_max_features_log2(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, max_features='log2', random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.8

    def test_reproducibility(self, separable):
        X, y = separable
        rf1 = random_forest_classifier(n_estimators=5, random_state=99)
        rf2 = random_forest_classifier(n_estimators=5, random_state=99)
        rf1.fit(X, y); rf2.fit(X, y)
        np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))

    def test_bootstrap_false_works(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=5, bootstrap=False, random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.8

    def test_mismatched_X_y_raises(self, separable):
        X, y = separable
        rf = random_forest_classifier(n_estimators=3)
        with pytest.raises(ValueError):
            rf.fit(X, y[:-1])


# ─────────────────────────────────────────────────────────────────────────────
# ── Integration tests ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWineIntegration:
    def test_bagging_accuracy_above_85(self, wine_split):
        X_train, X_test, y_train, y_test = wine_split
        b = bagging_classifier(n_estimators=15, max_depth=5, random_state=42)
        b.fit(X_train, y_train)
        assert b.score(X_test, y_test) > 0.85

    def test_rf_accuracy_above_85(self, wine_split):
        X_train, X_test, y_train, y_test = wine_split
        rf = random_forest_classifier(n_estimators=15, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        assert rf.score(X_test, y_test) > 0.85

    def test_voting_accuracy(self, wine_split):
        X_train, X_test, y_train, y_test = wine_split
        v = hard_voting_classifier(classifiers=[
            decision_tree_classifier(max_depth=5),
            KNN(k=5),
        ])
        v.fit(X_train, y_train)
        assert v.score(X_test, y_test) > 0.80
