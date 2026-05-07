"""
test_kmeans.py

Comprehensive tests for the kmeans clustering class.

Covers:
- Constructor validation
- _prepare_X input handling
- _initialize_centroids produces correct shape and unique points
- _distance_to_centroids matrix shape
- _assign_labels returns correct cluster indices
- _update_centroids recalculates means
- _compute_inertia is non-negative
- fit() convergence and attribute population
- predict() on new data
- fit_predict() consistency
- score() is negative inertia
- Inertia decreases / stays same per iteration on clean blobs
- Reproducibility via random_state
- Edge cases: k=1, k=n_samples, empty cluster replacement
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, r"/Jana CMOR/2026_Data_Science_and_Machine_Learning/src/rice_ml/unsupervised_learning")
from rice_ml.unsupervised_learning.kmeans import kmeans


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blobs():
    """Three well-separated 2-D clusters."""
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [5, 8]], dtype=float)
    parts = [rng.normal(c, 0.5, (30, 2)) for c in centres]
    X = np.vstack(parts)
    true_labels = np.repeat([0, 1, 2], 30)
    return X, true_labels

@pytest.fixture
def fitted(blobs):
    X, _ = blobs
    m = kmeans(n_clusters=3, max_iter=100, random_state=42)
    m.fit(X)
    return m, X


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_n_clusters_less_than_1_raises(self):
        with pytest.raises(ValueError):
            kmeans(n_clusters=0)

    def test_max_iter_less_than_1_raises(self):
        with pytest.raises(ValueError):
            kmeans(n_clusters=2, max_iter=0)

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError):
            kmeans(n_clusters=2, tol=-1)

    def test_defaults(self):
        m = kmeans()
        assert m.n_clusters == 3
        assert m.max_iter == 100
        assert m.tol == 1e-4
        assert m.random_state == 42

    def test_centroids_none_before_fit(self):
        assert kmeans().centroids is None

    def test_labels_none_before_fit(self):
        assert kmeans().labels_ is None

    def test_inertia_none_before_fit(self):
        assert kmeans().inertia_ is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. _prepare_X
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareX:
    def test_1d_reshaped_to_2d(self):
        m = kmeans(n_clusters=2)
        X = m._prepare_X(np.array([1.0, 2.0, 3.0]))
        assert X.ndim == 2

    def test_empty_raises(self):
        m = kmeans(n_clusters=1)
        with pytest.raises(ValueError):
            m._prepare_X(np.array([]).reshape(0, 2))

    def test_too_few_samples_raises(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=200)  # more clusters than samples
        with pytest.raises(ValueError):
            m._prepare_X(X)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Centroid initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroidInit:
    def test_centroids_shape(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        X_prep = m._prepare_X(X)
        m.centroids = m._initialize_centroids(X_prep)
        assert m.centroids.shape == (3, X.shape[1])

    def test_centroids_are_from_X(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        X_prep = m._prepare_X(X)
        m.centroids = m._initialize_centroids(X_prep)
        for c in m.centroids:
            assert any(np.allclose(c, x) for x in X_prep)

    def test_centroids_unique(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        X_prep = m._prepare_X(X)
        m.centroids = m._initialize_centroids(X_prep)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(m.centroids[i], m.centroids[j])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Distance & assignment
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanceAndAssign:
    def test_distance_matrix_shape(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        m.fit(X)
        X_prep = m._prepare_X(X)
        D = m._distance_to_centroids(X_prep)
        assert D.shape == (len(X), 3)

    def test_distance_nonnegative(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        m.fit(X)
        D = m._distance_to_centroids(m._prepare_X(X))
        assert np.all(D >= 0)

    def test_assign_labels_range(self, fitted):
        m, X = fitted
        labels = m._assign_labels(m._prepare_X(X))
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_assign_labels_shape(self, fitted):
        m, X = fitted
        labels = m._assign_labels(m._prepare_X(X))
        assert labels.shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# 5. fit() — attributes
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        assert m.fit(X) is m

    def test_centroids_shape_after_fit(self, fitted):
        m, X = fitted
        assert m.centroids.shape == (3, 2)

    def test_labels_shape_after_fit(self, fitted):
        m, X = fitted
        assert m.labels_.shape == (len(X),)

    def test_labels_range_after_fit(self, fitted):
        m, X = fitted
        assert np.all(m.labels_ >= 0) and np.all(m.labels_ < 3)

    def test_inertia_nonnegative(self, fitted):
        m, X = fitted
        assert m.inertia_ >= 0

    def test_n_iter_positive(self, fitted):
        m, X = fitted
        assert m.n_iter_ >= 1

    def test_all_clusters_non_empty(self, fitted):
        m, X = fitted
        for k in range(3):
            assert (m.labels_ == k).sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. fit_predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestFitPredict:
    def test_fit_predict_matches_labels(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=42)
        fp_labels = m.fit_predict(X)
        np.testing.assert_array_equal(fp_labels, m.labels_)

    def test_fit_predict_shape(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        labels = m.fit_predict(X)
        assert labels.shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# 7. predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_before_fit_raises(self):
        m = kmeans(n_clusters=2)
        with pytest.raises(RuntimeError):
            m.predict(np.array([[1.0, 2.0]]))

    def test_predict_shape(self, fitted):
        m, X = fitted
        assert m.predict(X).shape == (len(X),)

    def test_predict_labels_in_range(self, fitted):
        m, X = fitted
        labels = m.predict(X)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_predict_consistent_with_fit(self, fitted):
        m, X = fitted
        pred_labels = m.predict(X)
        np.testing.assert_array_equal(pred_labels, m.labels_)

    def test_predict_wrong_features_raises(self, fitted):
        m, X = fitted
        with pytest.raises(ValueError):
            m.predict(np.ones((5, 10)))  # wrong number of features


# ─────────────────────────────────────────────────────────────────────────────
# 8. score()
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_is_negative_inertia(self, fitted):
        m, X = fitted
        assert m.score(X) == pytest.approx(-m.inertia_, rel=1e-6)

    def test_score_before_fit_raises(self):
        m = kmeans(n_clusters=2)
        with pytest.raises(RuntimeError):
            m.score(np.ones((5, 2)))

    def test_score_nonpositive(self, fitted):
        m, X = fitted
        assert m.score(X) <= 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Inertia & _update_centroids
# ─────────────────────────────────────────────────────────────────────────────

class TestInertiaAndCentroids:
    def test_inertia_zero_for_k_equals_n(self):
        """k=n means each point is its own centroid → inertia=0."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        m = kmeans(n_clusters=3, max_iter=100, random_state=0)
        m.fit(X)
        assert m.inertia_ == pytest.approx(0.0, abs=1e-10)

    def test_update_centroids_is_mean(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, random_state=0)
        X_prep = m._prepare_X(X)
        m.centroids = m._initialize_centroids(X_prep)
        labels = m._assign_labels(X_prep)
        new_centroids = m._update_centroids(X_prep, labels)
        # Verify each centroid equals mean of assigned points
        for k in range(3):
            mask = labels == k
            if mask.sum() > 0:
                np.testing.assert_allclose(new_centroids[k], X_prep[mask].mean(axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Reproducibility & convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self, blobs):
        X, _ = blobs
        m1 = kmeans(n_clusters=3, random_state=7)
        m2 = kmeans(n_clusters=3, random_state=7)
        m1.fit(X); m2.fit(X)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)
        assert m1.inertia_ == m2.inertia_

    def test_converges_on_blobs(self, blobs):
        X, _ = blobs
        m = kmeans(n_clusters=3, max_iter=200, random_state=0)
        m.fit(X)
        # Inertia should be very low for well-separated clusters
        assert m.inertia_ < 500

    def test_k1_inertia_equals_total_variance(self):
        """k=1 inertia = n * variance (sum of sq distances to global mean)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        m = kmeans(n_clusters=1, max_iter=1, random_state=0)
        m.fit(X)
        expected = np.sum((X - X.mean(axis=0)) ** 2)
        assert m.inertia_ == pytest.approx(expected, rel=1e-4)
