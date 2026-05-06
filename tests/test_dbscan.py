"""
test_dbscan.py

Comprehensive tests for the dbscan clustering class.

Covers:
- Constructor validation (eps, min_samples)
- Attribute state before and after fit
- _prepare_X input handling (1D, 2D, empty, 3D)
- _euclidean_distances_from_point correctness
- _find_neighbors correctness (eps boundary inclusion)
- _expand_cluster cluster growth logic
- fit() — labels_, core_sample_indices_, n_clusters_ population
- Point classification: core, border, and noise points
- fit_predict() consistency with fit()
- All-noise scenario (n_clusters_=0)
- All-one-cluster scenario
- Multi-cluster separation
- Correct noise label (-1) assignment
- Determinism (same result on repeated calls)
- Integration on make_moons and make_blobs
- sklearn label equivalence on identical parameters
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dbscan import dbscan


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_cluster_data():
    """Two tight clusters plus one isolated noise point."""
    return np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],    # cluster 0
        [5.0, 5.0],
        [5.1, 5.1],    # cluster 1
        [9.0, 9.0],    # noise
    ])

@pytest.fixture
def three_cluster_data():
    """Three well-separated 2-D blobs."""
    rng = np.random.default_rng(0)
    centres = [[0, 0], [8, 0], [4, 7]]
    parts = [rng.normal(c, 0.3, (20, 2)) for c in centres]
    return np.vstack(parts)

@pytest.fixture
def moons_data():
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
    return StandardScaler().fit_transform(X)

@pytest.fixture
def blobs_data():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.4, random_state=0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_eps(self):
        assert dbscan().eps == 0.5

    def test_default_min_samples(self):
        assert dbscan().min_samples == 5

    def test_custom_eps(self):
        assert dbscan(eps=1.5).eps == 1.5

    def test_custom_min_samples(self):
        assert dbscan(min_samples=3).min_samples == 3

    def test_eps_zero_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            dbscan(eps=0)

    def test_eps_negative_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            dbscan(eps=-0.1)

    def test_min_samples_zero_raises(self):
        with pytest.raises(ValueError, match="min_samples must be at least 1"):
            dbscan(min_samples=0)

    def test_min_samples_negative_raises(self):
        with pytest.raises(ValueError, match="min_samples must be at least 1"):
            dbscan(min_samples=-1)

    def test_labels_none_before_fit(self):
        assert dbscan().labels_ is None

    def test_core_sample_indices_none_before_fit(self):
        assert dbscan().core_sample_indices_ is None

    def test_n_clusters_zero_before_fit(self):
        assert dbscan().n_clusters_ == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. _prepare_X input handling
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareX:
    def test_2d_input_returned_unchanged_shape(self):
        m = dbscan()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = m._prepare_X(X)
        assert result.shape == (2, 2)

    def test_1d_input_reshaped_to_column(self):
        m = dbscan()
        result = m._prepare_X(np.array([1.0, 2.0, 3.0]))
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_empty_array_raises(self):
        m = dbscan()
        with pytest.raises(ValueError, match="at least one sample"):
            m._prepare_X(np.array([]).reshape(0, 2))

    def test_3d_input_raises(self):
        m = dbscan()
        with pytest.raises(ValueError):
            m._prepare_X(np.ones((2, 2, 2)))

    def test_list_input_converted(self):
        m = dbscan()
        result = m._prepare_X([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_integer_array_cast_to_float(self):
        m = dbscan()
        result = m._prepare_X(np.array([[1, 2], [3, 4]]))
        assert result.dtype == float


# ─────────────────────────────────────────────────────────────────────────────
# 3. _euclidean_distances_from_point
# ─────────────────────────────────────────────────────────────────────────────

class TestEuclideanDistances:
    def test_distance_to_self_is_zero(self):
        m = dbscan()
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        dists = m._euclidean_distances_from_point(X, 0)
        assert dists[0] == pytest.approx(0.0)

    def test_3_4_5_triangle(self):
        m = dbscan()
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        dists = m._euclidean_distances_from_point(X, 0)
        assert dists[1] == pytest.approx(5.0)

    def test_distances_nonnegative(self):
        m = dbscan()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        dists = m._euclidean_distances_from_point(X, 0)
        assert np.all(dists >= 0)

    def test_correct_shape(self):
        m = dbscan()
        X = np.random.default_rng(1).normal(size=(15, 4))
        dists = m._euclidean_distances_from_point(X, 0)
        assert dists.shape == (15,)

    def test_symmetry(self):
        m = dbscan()
        X = np.array([[1.0, 2.0], [4.0, 6.0], [0.0, 0.0]])
        d01 = m._euclidean_distances_from_point(X, 0)[1]
        d10 = m._euclidean_distances_from_point(X, 1)[0]
        assert d01 == pytest.approx(d10)

    def test_known_unit_vector(self):
        m = dbscan()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dists = m._euclidean_distances_from_point(X, 0)
        assert dists[1] == pytest.approx(1.0)
        assert dists[2] == pytest.approx(1.0)

    def test_high_dimensional(self):
        m = dbscan()
        X = np.zeros((3, 100))
        X[1] = 1.0  # distance from X[0] = sqrt(100) = 10
        dists = m._euclidean_distances_from_point(X, 0)
        assert dists[1] == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _find_neighbors
# ─────────────────────────────────────────────────────────────────────────────

class TestFindNeighbors:
    def test_point_always_its_own_neighbor(self):
        """Every point is within eps of itself."""
        m = dbscan(eps=0.5)
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        nbrs = m._find_neighbors(X, 0)
        assert 0 in nbrs

    def test_close_point_included(self):
        m = dbscan(eps=1.0)
        X = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]])
        nbrs = m._find_neighbors(X, 0)
        assert 1 in nbrs

    def test_far_point_excluded(self):
        m = dbscan(eps=1.0)
        X = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]])
        nbrs = m._find_neighbors(X, 0)
        assert 2 not in nbrs

    def test_eps_boundary_included(self):
        """Point exactly at eps distance must be included."""
        m = dbscan(eps=1.0)
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        nbrs = m._find_neighbors(X, 0)
        assert 1 in nbrs

    def test_returns_ndarray(self):
        m = dbscan(eps=1.0)
        X = np.array([[0.0, 0.0], [0.5, 0.0]])
        assert isinstance(m._find_neighbors(X, 0), np.ndarray)

    def test_all_close_returns_all(self):
        m = dbscan(eps=10.0)
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        nbrs = m._find_neighbors(X, 0)
        assert len(nbrs) == 3

    def test_isolated_returns_only_self(self):
        m = dbscan(eps=0.1)
        X = np.array([[0.0, 0.0], [100.0, 0.0]])
        nbrs = m._find_neighbors(X, 0)
        assert list(nbrs) == [0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. fit() — attribute population
# ─────────────────────────────────────────────────────────────────────────────

class TestFitAttributes:
    def test_fit_returns_self(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        assert m.fit(two_cluster_data) is m

    def test_labels_set_after_fit(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert m.labels_ is not None

    def test_labels_shape(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert m.labels_.shape == (len(two_cluster_data),)

    def test_labels_dtype_int(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert np.issubdtype(m.labels_.dtype, np.integer)

    def test_core_sample_indices_set_after_fit(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert m.core_sample_indices_ is not None

    def test_core_sample_indices_dtype_int(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert np.issubdtype(m.core_sample_indices_.dtype, np.integer)

    def test_core_sample_indices_sorted(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert list(m.core_sample_indices_) == sorted(m.core_sample_indices_)

    def test_core_sample_indices_in_valid_range(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        n = len(two_cluster_data)
        assert np.all(m.core_sample_indices_ >= 0)
        assert np.all(m.core_sample_indices_ < n)

    def test_n_clusters_set_after_fit(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        assert m.n_clusters_ == 2

    def test_n_clusters_equals_unique_non_noise_labels(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        expected = len(set(m.labels_) - {-1})
        assert m.n_clusters_ == expected


# ─────────────────────────────────────────────────────────────────────────────
# 6. Point classification — core, border, noise
# ─────────────────────────────────────────────────────────────────────────────

class TestPointClassification:
    def test_noise_points_labeled_minus_one(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        # Point index 5 at [9,9] should be noise
        assert m.labels_[5] == -1

    def test_cluster_labels_nonnegative_for_non_noise(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        non_noise_labels = m.labels_[m.labels_ != -1]
        assert np.all(non_noise_labels >= 0)

    def test_core_points_in_clusters(self, two_cluster_data):
        """All core point indices must not be labeled as noise."""
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        for idx in m.core_sample_indices_:
            assert m.labels_[idx] != -1

    def test_all_noise_when_eps_tiny(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                      [3.0, 0.0], [4.0, 0.0]])
        m = dbscan(eps=0.01, min_samples=2)
        m.fit(X)
        assert np.all(m.labels_ == -1)

    def test_n_clusters_zero_when_all_noise(self):
        X = np.eye(5) * 100
        m = dbscan(eps=0.1, min_samples=2)
        m.fit(X)
        assert m.n_clusters_ == 0

    def test_core_sample_indices_empty_when_all_noise(self):
        X = np.eye(5) * 100
        m = dbscan(eps=0.1, min_samples=2)
        m.fit(X)
        assert len(m.core_sample_indices_) == 0

    def test_border_point_not_core_but_in_cluster(self):
        """
        Layout: core=[0,1,2] tightly packed, point 3 only near point 2.
        min_samples=3 means point 3 has only 2 neighbours → border point.
        """
        X = np.array([[0.0, 0.0],
                      [0.1, 0.0],
                      [0.2, 0.0],  # core
                      [0.65, 0.0]]) # border: within eps of [2] but not a core
        m = dbscan(eps=0.5, min_samples=3)
        m.fit(X)
        # Point 3 should be in a cluster (border), not noise
        assert m.labels_[3] != -1
        # Point 3 should NOT be a core point
        assert 3 not in m.core_sample_indices_

    def test_single_point_min_samples_1_is_core(self):
        m = dbscan(eps=0.5, min_samples=1)
        m.fit(np.array([[0.0, 0.0]]))
        assert m.labels_[0] == 0
        assert m.n_clusters_ == 1
        assert 0 in m.core_sample_indices_

    def test_single_point_min_samples_2_is_noise(self):
        m = dbscan(eps=0.5, min_samples=2)
        m.fit(np.array([[0.0, 0.0]]))
        assert m.labels_[0] == -1
        assert m.n_clusters_ == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cluster label properties
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterLabels:
    def test_labels_start_from_zero(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        non_noise = m.labels_[m.labels_ != -1]
        assert non_noise.min() == 0

    def test_cluster_ids_contiguous(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        cluster_ids = sorted(set(m.labels_) - {-1})
        assert cluster_ids == list(range(m.n_clusters_))

    def test_two_clusters_different_labels(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        label_c0 = m.labels_[0]  # first cluster
        label_c1 = m.labels_[3]  # second cluster
        assert label_c0 != label_c1
        assert label_c0 != -1
        assert label_c1 != -1

    def test_same_cluster_same_label(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(two_cluster_data)
        # Indices 0,1,2 form cluster 0
        assert m.labels_[0] == m.labels_[1] == m.labels_[2]

    def test_known_two_cluster_result(self):
        """Exact label check matching docstring example."""
        X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
                      [5.0, 5.0], [5.1, 5.1], [9.0, 9.0]])
        m = dbscan(eps=0.35, min_samples=2)
        m.fit(X)
        assert m.labels_[5] == -1          # noise
        assert m.labels_[0] == m.labels_[1] == m.labels_[2]  # cluster 0
        assert m.labels_[3] == m.labels_[4]                   # cluster 1
        assert m.labels_[0] != m.labels_[3]                   # different clusters

    def test_all_one_cluster_when_eps_large(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        m = dbscan(eps=10.0, min_samples=1)
        m.fit(X)
        assert m.n_clusters_ == 1
        assert np.all(m.labels_ == 0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. fit_predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestFitPredict:
    def test_fit_predict_returns_ndarray(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        labels = m.fit_predict(two_cluster_data)
        assert isinstance(labels, np.ndarray)

    def test_fit_predict_matches_labels_(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        labels = m.fit_predict(two_cluster_data)
        np.testing.assert_array_equal(labels, m.labels_)

    def test_fit_predict_shape(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        labels = m.fit_predict(two_cluster_data)
        assert labels.shape == (len(two_cluster_data),)

    def test_fit_predict_populates_n_clusters(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit_predict(two_cluster_data)
        assert m.n_clusters_ == 2

    def test_fit_predict_populates_core_indices(self, two_cluster_data):
        m = dbscan(eps=0.35, min_samples=2)
        m.fit_predict(two_cluster_data)
        assert m.core_sample_indices_ is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9. 1D input
# ─────────────────────────────────────────────────────────────────────────────

class Test1DInput:
    def test_1d_two_clusters(self):
        X = np.array([1.0, 1.1, 1.2, 9.0, 9.1])
        m = dbscan(eps=0.5, min_samples=2)
        m.fit(X)
        assert m.n_clusters_ == 2

    def test_1d_labels_shape(self):
        X = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
        m = dbscan(eps=0.3, min_samples=2)
        m.fit(X)
        assert m.labels_.shape == (5,)

    def test_1d_noise(self):
        X = np.array([0.0, 0.1, 50.0])
        m = dbscan(eps=0.2, min_samples=2)
        m.fit(X)
        assert m.labels_[2] == -1


# ─────────────────────────────────────────────────────────────────────────────
# 10. Three clusters
# ─────────────────────────────────────────────────────────────────────────────

class TestThreeClusters:
    def test_three_clusters_found(self, three_cluster_data):
        m = dbscan(eps=1.0, min_samples=5)
        m.fit(three_cluster_data)
        assert m.n_clusters_ == 3

    def test_all_points_assigned(self, three_cluster_data):
        m = dbscan(eps=1.0, min_samples=5)
        m.fit(three_cluster_data)
        assert np.all(m.labels_ >= 0)  # no noise for tightly packed blobs

    def test_labels_shape(self, three_cluster_data):
        m = dbscan(eps=1.0, min_samples=5)
        m.fit(three_cluster_data)
        assert m.labels_.shape == (len(three_cluster_data),)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_result_two_calls(self, two_cluster_data):
        m1 = dbscan(eps=0.35, min_samples=2)
        m2 = dbscan(eps=0.35, min_samples=2)
        m1.fit(two_cluster_data)
        m2.fit(two_cluster_data)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_same_result_fit_then_fit_predict(self, two_cluster_data):
        m1 = dbscan(eps=0.35, min_samples=2)
        m2 = dbscan(eps=0.35, min_samples=2)
        m1.fit(two_cluster_data)
        m2.fit_predict(two_cluster_data)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)


# ─────────────────────────────────────────────────────────────────────────────
# 12. eps and min_samples sensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterSensitivity:
    def test_larger_eps_merges_clusters(self):
        """Two tight clusters that are clearly separate: small eps=2 finds 2 clusters,
        large eps=20 merges them into 1."""
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal([0, 0], 0.3, (15, 2)),
            rng.normal([10, 0], 0.3, (15, 2)),
        ])
        m_small = dbscan(eps=2.0, min_samples=3)
        m_small.fit(X)
        m_large = dbscan(eps=20.0, min_samples=3)
        m_large.fit(X)
        assert m_small.n_clusters_ == 2
        assert m_large.n_clusters_ == 1
        assert m_large.n_clusters_ <= m_small.n_clusters_

    def test_larger_min_samples_more_noise(self, two_cluster_data):
        """Requiring more neighbors means more noise points."""
        m_low  = dbscan(eps=0.35, min_samples=2)
        m_high = dbscan(eps=0.35, min_samples=10)
        m_low.fit(two_cluster_data)
        m_high.fit(two_cluster_data)
        noise_low  = (m_low.labels_  == -1).sum()
        noise_high = (m_high.labels_ == -1).sum()
        assert noise_high >= noise_low

    def test_min_samples_1_no_noise(self):
        """With min_samples=1 every point is a core point — no noise."""
        X = np.random.default_rng(0).normal(size=(30, 2))
        m = dbscan(eps=0.5, min_samples=1)
        m.fit(X)
        assert np.all(m.labels_ >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Integration — make_moons (matches sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class TestMoonsIntegration:
    def test_n_clusters_matches_sklearn(self, moons_data):
        import sklearn.cluster as skc
        eps, ms = 0.3, 5
        my = dbscan(eps=eps, min_samples=ms)
        sk = skc.DBSCAN(eps=eps, min_samples=ms)
        my.fit(moons_data)
        sk.fit(moons_data)
        assert my.n_clusters_ == len(set(sk.labels_) - {-1})

    def test_noise_count_matches_sklearn(self, moons_data):
        import sklearn.cluster as skc
        eps, ms = 0.3, 5
        my = dbscan(eps=eps, min_samples=ms)
        sk = skc.DBSCAN(eps=eps, min_samples=ms)
        my.fit(moons_data)
        sk.fit(moons_data)
        assert (my.labels_ == -1).sum() == (sk.labels_ == -1).sum()

    def test_labels_identical_to_sklearn(self, moons_data):
        import sklearn.cluster as skc
        eps, ms = 0.3, 5
        my = dbscan(eps=eps, min_samples=ms)
        sk = skc.DBSCAN(eps=eps, min_samples=ms)
        my.fit(moons_data)
        sk.fit(moons_data)
        np.testing.assert_array_equal(my.labels_, sk.labels_)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Integration — make_blobs (3 clusters, no noise)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlobsIntegration:
    def test_finds_three_clusters(self, blobs_data):
        m = dbscan(eps=0.8, min_samples=5)
        m.fit(blobs_data)
        assert m.n_clusters_ == 3

    def test_no_noise_on_clean_blobs(self, blobs_data):
        m = dbscan(eps=0.8, min_samples=5)
        m.fit(blobs_data)
        assert (m.labels_ == -1).sum() == 0

    def test_all_150_points_labelled(self, blobs_data):
        m = dbscan(eps=0.8, min_samples=5)
        m.fit(blobs_data)
        assert m.labels_.shape == (150,)

    def test_core_indices_nonempty_on_blobs(self, blobs_data):
        m = dbscan(eps=0.8, min_samples=5)
        m.fit(blobs_data)
        assert len(m.core_sample_indices_) > 0
