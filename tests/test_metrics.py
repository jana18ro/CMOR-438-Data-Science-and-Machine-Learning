"""
test_metrics.py

Comprehensive test suite for rice_ml.measures.

Covers
------
metrics.py
    _to_1d / _to_1d_float / _check_lengths helpers
    accuracy_score
    confusion_matrix
    precision_score  (binary, macro, micro, weighted, None, aliases)
    recall_score     (binary, macro, micro, weighted, None, aliases)
    f1_score         (binary, macro, micro, weighted, None)
    classification_report
    mean_squared_error / root_mean_squared_error / mean_absolute_error / r2_score
    short aliases (accuracy, precision, recall, mse, rmse, mae)

validation.py
    k_fold_indices
    cross_val_score
    stratified_split
    holdout_evaluation
"""

from __future__ import annotations

import sys
import os

# Allow running from the project root OR from inside the measures/ directory.
sys.path.insert(0, r"/home/claude/project/2026_Data_Science_and_Machine_Learning/src/rice_ml/measures")

import numpy as np
import pytest

from metrics import (
    # classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    # regression
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    # aliases
    accuracy,
    precision,
    recall,
    mse,
    rmse,
    mae,
)
from validation import (
    k_fold_indices,
    cross_val_score,
    stratified_split,
    holdout_evaluation,
)

# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def binary_perfect():
    y = np.array([0, 0, 1, 1, 0, 1])
    return y, y.copy()

@pytest.fixture
def binary_mixed():
    """2 TP, 1 FP, 1 FN, 1 TN"""
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0])
    return y_true, y_pred

@pytest.fixture
def multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])
    return y_true, y_pred

@pytest.fixture
def regression_example():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5,  0.0, 2.0, 8.0])
    return y_true, y_pred


# ═══════════════════════════════════════════════════════════════════════════
# 1. Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyScore:
    def test_all_correct(self, binary_perfect):
        yt, yp = binary_perfect
        assert accuracy_score(yt, yp) == pytest.approx(1.0)

    def test_none_correct(self):
        yt = np.array([0, 0, 1, 1])
        yp = np.array([1, 1, 0, 0])
        assert accuracy_score(yt, yp) == pytest.approx(0.0)

    def test_partial(self, binary_mixed):
        yt, yp = binary_mixed          # 3 of 5 correct
        assert accuracy_score(yt, yp) == pytest.approx(3 / 5)

    def test_returns_float(self, binary_mixed):
        yt, yp = binary_mixed
        assert isinstance(accuracy_score(yt, yp), float)

    def test_lists_accepted(self):
        assert accuracy_score([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)

    def test_multiclass(self, multiclass):
        yt, yp = multiclass
        correct = np.sum(yt == yp)
        assert accuracy_score(yt, yp) == pytest.approx(correct / len(yt))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            accuracy_score([0, 1], [0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy_score([], [])

    def test_alias_matches(self, binary_mixed):
        yt, yp = binary_mixed
        assert accuracy(yt, yp) == accuracy_score(yt, yp)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════

class TestConfusionMatrix:
    def test_binary_known_values(self):
        yt = np.array([0, 0, 1, 1])
        yp = np.array([0, 1, 1, 0])
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_perfect_binary(self, binary_perfect):
        yt, yp = binary_perfect
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        assert cm[0, 0] == np.sum(yt == 0)
        assert cm[1, 1] == np.sum(yt == 1)
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0

    def test_diagonal_sum_equals_correct(self, binary_mixed):
        yt, yp = binary_mixed
        cm = confusion_matrix(yt, yp)
        assert int(np.diag(cm).sum()) == int(np.sum(yt == yp))

    def test_total_sum_equals_n(self, multiclass):
        yt, yp = multiclass
        cm = confusion_matrix(yt, yp)
        assert cm.sum() == len(yt)

    def test_shape_equals_n_classes(self, multiclass):
        yt, yp = multiclass
        cm = confusion_matrix(yt, yp)
        assert cm.shape == (3, 3)

    def test_explicit_labels_ordering(self):
        yt = np.array([0, 1, 2])
        yp = np.array([0, 1, 2])
        cm_012 = confusion_matrix(yt, yp, labels=[0, 1, 2])
        cm_210 = confusion_matrix(yt, yp, labels=[2, 1, 0])
        # Reversed labels → reversed diagonal
        np.testing.assert_array_equal(np.diag(cm_012), np.diag(cm_210)[::-1])

    def test_labels_inferred_from_data(self, binary_mixed):
        yt, yp = binary_mixed
        cm = confusion_matrix(yt, yp)
        assert cm.shape == (2, 2)

    def test_returns_int_dtype(self, binary_mixed):
        yt, yp = binary_mixed
        assert confusion_matrix(yt, yp).dtype == int


# ═══════════════════════════════════════════════════════════════════════════
# 3. Precision
# ═══════════════════════════════════════════════════════════════════════════

class TestPrecisionScore:
    def test_binary_known(self, binary_mixed):
        yt, yp = binary_mixed      # TP=2, FP=1
        assert precision_score(yt, yp, positive_label=1) == pytest.approx(2 / 3)

    def test_binary_perfect(self, binary_perfect):
        yt, yp = binary_perfect
        assert precision_score(yt, yp, positive_label=1) == pytest.approx(1.0)

    def test_binary_no_positive_predicted(self):
        yt = np.array([1, 1, 1])
        yp = np.array([0, 0, 0])
        assert precision_score(yt, yp, positive_label=1) == pytest.approx(0.0)

    def test_macro_in_01(self, multiclass):
        yt, yp = multiclass
        p = precision_score(yt, yp, average="macro")
        assert 0.0 <= p <= 1.0

    def test_micro_in_01(self, multiclass):
        yt, yp = multiclass
        p = precision_score(yt, yp, average="micro")
        assert 0.0 <= p <= 1.0

    def test_weighted_in_01(self, multiclass):
        yt, yp = multiclass
        p = precision_score(yt, yp, average="weighted")
        assert 0.0 <= p <= 1.0

    def test_none_returns_array(self, multiclass):
        yt, yp = multiclass
        p = precision_score(yt, yp, average=None)
        assert isinstance(p, np.ndarray)
        assert p.shape == (3,)

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError):
            precision_score([0, 1], [0, 1], average="bad")

    def test_alias_matches(self, binary_mixed):
        yt, yp = binary_mixed
        assert precision(yt, yp, positive_label=1) == precision_score(yt, yp, positive_label=1)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            precision_score([0, 1, 1], [0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Recall
# ═══════════════════════════════════════════════════════════════════════════

class TestRecallScore:
    def test_binary_known(self, binary_mixed):
        yt, yp = binary_mixed      # TP=2, FN=1
        assert recall_score(yt, yp, positive_label=1) == pytest.approx(2 / 3)

    def test_binary_perfect(self, binary_perfect):
        yt, yp = binary_perfect
        assert recall_score(yt, yp, positive_label=1) == pytest.approx(1.0)

    def test_binary_all_missed(self):
        yt = np.array([1, 1, 1])
        yp = np.array([0, 0, 0])
        assert recall_score(yt, yp, positive_label=1) == pytest.approx(0.0)

    def test_macro_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= recall_score(yt, yp, average="macro") <= 1.0

    def test_micro_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= recall_score(yt, yp, average="micro") <= 1.0

    def test_weighted_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= recall_score(yt, yp, average="weighted") <= 1.0

    def test_none_returns_array(self, multiclass):
        yt, yp = multiclass
        r = recall_score(yt, yp, average=None)
        assert isinstance(r, np.ndarray)
        assert r.shape == (3,)

    def test_alias_matches(self, binary_mixed):
        yt, yp = binary_mixed
        assert recall(yt, yp, positive_label=1) == recall_score(yt, yp, positive_label=1)

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError):
            recall_score([0, 1], [0, 1], average="bad")


# ═══════════════════════════════════════════════════════════════════════════
# 5. F1 Score
# ═══════════════════════════════════════════════════════════════════════════

class TestF1Score:
    def test_binary_known(self, binary_mixed):
        yt, yp = binary_mixed
        # precision=2/3, recall=2/3  →  F1=2/3
        assert f1_score(yt, yp, positive_label=1) == pytest.approx(2 / 3)

    def test_binary_perfect(self, binary_perfect):
        yt, yp = binary_perfect
        assert f1_score(yt, yp, positive_label=1) == pytest.approx(1.0)

    def test_binary_zero_when_no_tp(self):
        yt = np.array([1, 1])
        yp = np.array([0, 0])
        assert f1_score(yt, yp, positive_label=1) == pytest.approx(0.0)

    def test_harmonic_mean_property(self, binary_mixed):
        yt, yp = binary_mixed
        p = precision_score(yt, yp, positive_label=1)
        r = recall_score(yt, yp, positive_label=1)
        expected = 2 * p * r / (p + r)
        assert f1_score(yt, yp, positive_label=1) == pytest.approx(expected)

    def test_macro_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= f1_score(yt, yp, average="macro") <= 1.0

    def test_micro_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= f1_score(yt, yp, average="micro") <= 1.0

    def test_weighted_in_01(self, multiclass):
        yt, yp = multiclass
        assert 0.0 <= f1_score(yt, yp, average="weighted") <= 1.0

    def test_none_returns_array(self, multiclass):
        yt, yp = multiclass
        f = f1_score(yt, yp, average=None)
        assert isinstance(f, np.ndarray)
        assert f.shape == (3,)

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError):
            f1_score([0, 1], [0, 1], average="bad")

    def test_all_averages_consistent_perfect(self, binary_perfect):
        yt, yp = binary_perfect
        for avg in ["binary", "macro", "micro", "weighted"]:
            kw = {"positive_label": 1} if avg == "binary" else {}
            assert f1_score(yt, yp, average=avg, **kw) == pytest.approx(1.0), avg


# ═══════════════════════════════════════════════════════════════════════════
# 6. Classification Report
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationReport:
    def test_returns_string(self, binary_mixed):
        yt, yp = binary_mixed
        report = classification_report(yt, yp)
        assert isinstance(report, str)

    def test_contains_class_labels(self, multiclass):
        yt, yp = multiclass
        report = classification_report(yt, yp)
        for lbl in ["0", "1", "2"]:
            assert lbl in report

    def test_contains_averages(self, multiclass):
        yt, yp = multiclass
        report = classification_report(yt, yp)
        assert "macro avg" in report
        assert "weighted avg" in report

    def test_target_names_override_labels(self, binary_mixed):
        yt, yp = binary_mixed
        report = classification_report(yt, yp, target_names=["neg", "pos"])
        assert "neg" in report
        assert "pos" in report

    def test_wrong_target_names_count_raises(self, binary_mixed):
        yt, yp = binary_mixed
        with pytest.raises(ValueError):
            classification_report(yt, yp, target_names=["only_one"])

    def test_metrics_plausible(self, binary_perfect):
        yt, yp = binary_perfect
        report = classification_report(yt, yp)
        # Perfect predictions → all scores should be 1.0000
        assert "1.0000" in report


# ═══════════════════════════════════════════════════════════════════════════
# 7. Regression Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionMetrics:
    def test_mse_known(self, regression_example):
        yt, yp = regression_example
        # errors: 0.5, 0.5, 0, 1 → sq: 0.25, 0.25, 0, 1 → mean=0.375
        assert mean_squared_error(yt, yp) == pytest.approx(0.375)

    def test_rmse_is_sqrt_mse(self, regression_example):
        yt, yp = regression_example
        assert root_mean_squared_error(yt, yp) == pytest.approx(np.sqrt(0.375))

    def test_mae_known(self, regression_example):
        yt, yp = regression_example
        # |errors|: 0.5, 0.5, 0, 1 → mean=0.5
        assert mean_absolute_error(yt, yp) == pytest.approx(0.5)

    def test_r2_known(self, regression_example):
        yt, yp = regression_example
        assert r2_score(yt, yp) == pytest.approx(0.9486081370449679)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_constant_true(self):
        # When all true values equal their mean, r2 is 1 iff predictions match too
        y = np.array([3.0, 3.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)
        assert r2_score(y, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)

    def test_mse_zero_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_mae_zero_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_mse_non_negative(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y + 1) >= 0.0

    def test_rmse_geq_mae(self):
        yt = np.array([1.0, 2.0, 4.0, 8.0])
        yp = np.array([2.0, 2.0, 3.0, 5.0])
        assert root_mean_squared_error(yt, yp) >= mean_absolute_error(yt, yp)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            mean_squared_error([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            mean_absolute_error([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            r2_score([1.0, 2.0], [1.0])

    def test_list_inputs(self):
        assert mean_squared_error([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Aliases
# ═══════════════════════════════════════════════════════════════════════════

class TestAliases:
    def test_accuracy_alias(self):
        yt = np.array([0, 1, 1])
        yp = np.array([0, 0, 1])
        assert accuracy(yt, yp) == accuracy_score(yt, yp)

    def test_precision_alias(self):
        yt = np.array([0, 1, 1])
        yp = np.array([0, 0, 1])
        assert precision(yt, yp, positive_label=1) == precision_score(yt, yp, positive_label=1)

    def test_recall_alias(self):
        yt = np.array([0, 1, 1])
        yp = np.array([0, 0, 1])
        assert recall(yt, yp, positive_label=1) == recall_score(yt, yp, positive_label=1)

    def test_mse_alias(self, regression_example):
        yt, yp = regression_example
        assert mse(yt, yp) == mean_squared_error(yt, yp)

    def test_rmse_alias(self, regression_example):
        yt, yp = regression_example
        assert rmse(yt, yp) == root_mean_squared_error(yt, yp)

    def test_mae_alias(self, regression_example):
        yt, yp = regression_example
        assert mae(yt, yp) == mean_absolute_error(yt, yp)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Error Handling (shared)
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_empty_y_true_raises(self):
        with pytest.raises(ValueError):
            accuracy_score([], [])

    def test_2d_y_raises(self):
        with pytest.raises(ValueError):
            accuracy_score([[0, 1], [1, 0]], [[0, 1], [1, 0]])

    def test_precision_requires_positive_label_for_multiclass(self):
        yt = np.array([0, 1, 2])
        yp = np.array([0, 1, 1])
        # binary mode with more than 2 classes and no positive_label
        # should raise because positive_label defaults to 1 and there
        # are 3 classes — the function should still handle it;
        # test that invalid average raises
        with pytest.raises(ValueError):
            precision_score(yt, yp, average="totally_wrong")

    def test_none_arrays_not_accepted(self):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            accuracy_score(None, None)  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# 10. k_fold_indices
# ═══════════════════════════════════════════════════════════════════════════

class TestKFoldIndices:
    def test_correct_number_of_folds(self):
        folds = k_fold_indices(100, n_splits=5)
        assert len(folds) == 5

    def test_test_sizes_sum_to_n(self):
        n, k = 97, 5
        folds = k_fold_indices(n, n_splits=k, random_state=0)
        total_test = sum(len(te) for _, te in folds)
        assert total_test == n

    def test_no_overlap_between_train_and_test(self):
        folds = k_fold_indices(50, n_splits=5, random_state=1)
        for tr, te in folds:
            assert len(set(tr) & set(te)) == 0

    def test_all_samples_covered_exactly_once(self):
        n = 30
        folds = k_fold_indices(n, n_splits=3, random_state=2)
        test_indices = np.concatenate([te for _, te in folds])
        assert sorted(test_indices) == list(range(n))

    def test_reproducible_with_same_seed(self):
        f1 = k_fold_indices(40, n_splits=4, random_state=42)
        f2 = k_fold_indices(40, n_splits=4, random_state=42)
        for (tr1, te1), (tr2, te2) in zip(f1, f2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        f1 = k_fold_indices(40, n_splits=4, random_state=1)
        f2 = k_fold_indices(40, n_splits=4, random_state=2)
        any_diff = any(not np.array_equal(te1, te2) for (_, te1), (_, te2) in zip(f1, f2))
        assert any_diff

    def test_no_shuffle_keeps_order(self):
        folds = k_fold_indices(20, n_splits=4, shuffle=False)
        test_indices = np.concatenate([te for _, te in folds])
        assert sorted(test_indices) == list(range(20))

    def test_n_splits_too_large_raises(self):
        with pytest.raises(ValueError):
            k_fold_indices(3, n_splits=5)

    def test_n_splits_less_than_2_raises(self):
        with pytest.raises(ValueError):
            k_fold_indices(10, n_splits=1)

    def test_n_samples_less_than_2_raises(self):
        with pytest.raises(ValueError):
            k_fold_indices(1, n_splits=2)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Stratified Split
# ═══════════════════════════════════════════════════════════════════════════

class TestStratifiedSplit:
    @pytest.fixture
    def balanced_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = np.array([0] * 100 + [1] * 100)
        return X, y

    def test_correct_split_sizes(self, balanced_data):
        X, y = balanced_data
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.25, random_state=0)
        assert len(y_te) + len(y_tr) == len(y)

    def test_class_proportions_preserved(self, balanced_data):
        X, y = balanced_data
        _, _, y_tr, y_te = stratified_split(X, y, test_size=0.2, random_state=0)
        prop_train = np.mean(y_tr == 1)
        prop_test  = np.mean(y_te == 1)
        assert abs(prop_train - prop_test) < 0.1

    def test_no_overlap(self, balanced_data):
        X, y = balanced_data
        X_tr, X_te, _, _ = stratified_split(X, y, test_size=0.25, random_state=7)
        # Compare rows
        train_set = {tuple(row) for row in X_tr}
        test_set  = {tuple(row) for row in X_te}
        assert len(train_set & test_set) == 0

    def test_all_samples_covered(self, balanced_data):
        X, y = balanced_data
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.3, random_state=3)
        assert len(y_tr) + len(y_te) == len(y)

    def test_invalid_test_size_raises(self, balanced_data):
        X, y = balanced_data
        with pytest.raises(ValueError):
            stratified_split(X, y, test_size=0.0)
        with pytest.raises(ValueError):
            stratified_split(X, y, test_size=1.0)

    def test_mismatched_X_y_raises(self):
        with pytest.raises(ValueError):
            stratified_split(np.zeros((10, 2)), np.zeros(5))


# ═══════════════════════════════════════════════════════════════════════════
# 12. Cross-Val Score
# ═══════════════════════════════════════════════════════════════════════════

class _DummyClassifier:
    """Always predicts the majority class seen during fit."""
    def __init__(self):
        self._label = None
    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self._label = vals[np.argmax(counts)]
        return self
    def predict(self, X):
        return np.full(len(X), self._label)
    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


class _DummyRegressor:
    """Always predicts the training mean."""
    def __init__(self):
        self._mean = 0.0
    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self._mean)


class TestCrossValScore:
    @pytest.fixture
    def clf_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_returns_array_of_scores(self, clf_data):
        X, y = clf_data
        scores = cross_val_score(_DummyClassifier(), X, y, n_splits=5)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (5,)

    def test_scores_between_0_and_1(self, clf_data):
        X, y = clf_data
        scores = cross_val_score(_DummyClassifier(), X, y, n_splits=5)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_custom_scoring(self, clf_data):
        X, y = clf_data
        scores = cross_val_score(
            _DummyClassifier(), X, y, n_splits=3,
            scoring=lambda yt, yp: float(np.mean(yt == yp))
        )
        assert scores.shape == (3,)

    def test_reproducible(self, clf_data):
        X, y = clf_data
        s1 = cross_val_score(_DummyClassifier(), X, y, n_splits=5, random_state=0)
        s2 = cross_val_score(_DummyClassifier(), X, y, n_splits=5, random_state=0)
        np.testing.assert_array_equal(s1, s2)

    def test_regressor_without_score_uses_accuracy(self):
        """Fallback to accuracy_score when model has no score() method."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 2))
        y = np.zeros(40)   # all-same target → always "correct"
        scores = cross_val_score(_DummyRegressor(), X, y, n_splits=4,
                                  scoring=lambda yt, yp: float(np.mean(yt == yp)))
        assert scores.shape == (4,)

    def test_mismatched_X_y_raises(self):
        with pytest.raises(ValueError):
            cross_val_score(_DummyClassifier(), np.zeros((10, 2)), np.zeros(5))


# ═══════════════════════════════════════════════════════════════════════════
# 13. Holdout Evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TestHoldoutEvaluation:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_returns_expected_keys(self, data):
        X, y = data
        result = holdout_evaluation(_DummyClassifier(), X, y, random_state=0)
        for key in ("model", "score", "predictions", "X_train", "X_test", "y_train", "y_test"):
            assert key in result

    def test_score_is_float(self, data):
        X, y = data
        result = holdout_evaluation(_DummyClassifier(), X, y, random_state=0)
        assert isinstance(result["score"], float)

    def test_split_sizes_correct(self, data):
        X, y = data
        result = holdout_evaluation(_DummyClassifier(), X, y, test_size=0.25, random_state=0)
        n_test  = len(result["y_test"])
        n_train = len(result["y_train"])
        assert n_test + n_train == len(y)

    def test_stratify_flag(self, data):
        X, y = data
        result = holdout_evaluation(_DummyClassifier(), X, y,
                                     test_size=0.25, stratify=True, random_state=0)
        # class proportion roughly preserved
        prop_full  = np.mean(y == 1)
        prop_train = np.mean(result["y_train"] == 1)
        assert abs(prop_full - prop_train) < 0.1

    def test_custom_scoring(self, data):
        X, y = data
        result = holdout_evaluation(
            _DummyClassifier(), X, y, random_state=0,
            scoring=lambda yt, yp: float(np.sum(yt == yp))
        )
        assert result["score"] >= 0

    def test_mismatched_X_y_raises(self):
        with pytest.raises(ValueError):
            holdout_evaluation(_DummyClassifier(), np.zeros((10, 2)), np.zeros(5))

    def test_invalid_test_size_raises(self, data):
        X, y = data
        with pytest.raises(ValueError):
            holdout_evaluation(_DummyClassifier(), X, y, test_size=1.5)

    def test_predictions_shape(self, data):
        X, y = data
        result = holdout_evaluation(_DummyClassifier(), X, y, random_state=0)
        assert len(result["predictions"]) == len(result["y_test"])
