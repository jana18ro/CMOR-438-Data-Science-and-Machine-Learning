"""
test_label_propagation_community_detection.py

Comprehensive tests for label_propagation_community_detection.

Covers:
- Constructor validation (max_iter, random_state)
- Attribute state before and after fit
- _prepare_adjacency_matrix input handling
    (non-square, 1D, empty, negative weights, 3D)
- _neighbor_indices correctness (weighted and unweighted)
- _weighted_label_vote tie-breaking and weight handling
- _compress_labels consecutive relabelling
- fit() — labels_, n_iter_, n_communities_ population
- fit_predict() consistency with fit()
- Isolated nodes keep unique labels
- Fully connected clique → 1 community
- Two disconnected cliques → 2 communities
- Early stopping when labels converge
- n_iter_ bounded by max_iter
- Weighted adjacency: strong edges dominate votes
- Reproducibility via random_state
- Compressed labels are consecutive from 0
- Integration: Karate Club-like structured graph
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from label_propagation_community_detection import label_propagation_community_detection as LPCD


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_cliques():
    """Two disjoint triangles — should produce exactly 2 communities."""
    A = np.zeros((6, 6), dtype=float)
    A[0:3, 0:3] = 1 - np.eye(3)   # clique 0: nodes {0,1,2}
    A[3:6, 3:6] = 1 - np.eye(3)   # clique 1: nodes {3,4,5}
    return A

@pytest.fixture
def triangle():
    """Fully connected 3-node graph — single community."""
    return np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)

@pytest.fixture
def path_graph():
    """Linear chain: 0-1-2-3-4."""
    n = 5
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A

@pytest.fixture
def weighted_two_communities():
    """
    6-node graph: nodes {0,1,2} tightly connected (weight=10),
    nodes {3,4,5} tightly connected (weight=10),
    weak bridge between communities (weight=0.01).
    """
    A = np.zeros((6, 6), dtype=float)
    A[0:3, 0:3] = 10 * (1 - np.eye(3))
    A[3:6, 3:6] = 10 * (1 - np.eye(3))
    A[2, 3] = A[3, 2] = 0.01   # weak bridge
    return A

@pytest.fixture
def isolated_node_graph():
    """Nodes 0 and 1 connected; node 2 isolated."""
    return np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_max_iter(self):
        assert LPCD().max_iter == 100

    def test_default_random_state(self):
        assert LPCD().random_state is None

    def test_custom_max_iter(self):
        assert LPCD(max_iter=50).max_iter == 50

    def test_custom_random_state(self):
        assert LPCD(random_state=7).random_state == 7

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter must be at least 1"):
            LPCD(max_iter=0)

    def test_max_iter_negative_raises(self):
        with pytest.raises(ValueError):
            LPCD(max_iter=-5)

    def test_labels_none_before_fit(self):
        assert LPCD().labels_ is None

    def test_n_iter_zero_before_fit(self):
        assert LPCD().n_iter_ == 0

    def test_n_communities_zero_before_fit(self):
        assert LPCD().n_communities_ == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. _prepare_adjacency_matrix input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareAdjacencyMatrix:
    def _m(self):
        return LPCD(max_iter=5)

    def test_valid_square_accepted(self):
        m = self._m()
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = m._prepare_adjacency_matrix(A)
        np.testing.assert_array_equal(result, A)

    def test_dtype_cast_to_float(self):
        m = self._m()
        A = np.array([[0, 1], [1, 0]], dtype=int)
        result = m._prepare_adjacency_matrix(A)
        assert result.dtype == float

    def test_list_input_converted(self):
        m = self._m()
        result = m._prepare_adjacency_matrix([[0, 1], [1, 0]])
        assert isinstance(result, np.ndarray)

    def test_1d_raises(self):
        m = self._m()
        with pytest.raises(ValueError, match="2D adjacency matrix"):
            m._prepare_adjacency_matrix(np.array([1, 2, 3]))

    def test_3d_raises(self):
        m = self._m()
        with pytest.raises(ValueError):
            m._prepare_adjacency_matrix(np.ones((2, 2, 2)))

    def test_empty_raises(self):
        m = self._m()
        with pytest.raises(ValueError, match="at least one node"):
            m._prepare_adjacency_matrix(np.zeros((0, 0)))

    def test_non_square_raises(self):
        m = self._m()
        with pytest.raises(ValueError, match="square"):
            m._prepare_adjacency_matrix(np.ones((3, 2)))

    def test_negative_weights_raise(self):
        m = self._m()
        with pytest.raises(ValueError, match="nonnegative"):
            m._prepare_adjacency_matrix(np.array([[0, -1], [-1, 0]]))

    def test_zero_weights_allowed(self):
        m = self._m()
        A = np.zeros((3, 3))
        result = m._prepare_adjacency_matrix(A)
        assert result.shape == (3, 3)

    def test_1x1_matrix_accepted(self):
        m = self._m()
        result = m._prepare_adjacency_matrix(np.array([[0.0]]))
        assert result.shape == (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _neighbor_indices
# ─────────────────────────────────────────────────────────────────────────────

class TestNeighborIndices:
    def _m(self):
        return LPCD()

    def test_no_neighbors_for_isolated_node(self, isolated_node_graph):
        m = self._m()
        nbrs = m._neighbor_indices(isolated_node_graph, 2)
        assert len(nbrs) == 0

    def test_correct_neighbors_unweighted(self):
        m = self._m()
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        nbrs = m._neighbor_indices(A, 1)
        assert set(nbrs) == {0, 2}

    def test_no_self_loop_in_neighbors_without_diagonal(self):
        m = self._m()
        A = np.array([[0, 1], [1, 0]], dtype=float)
        nbrs = m._neighbor_indices(A, 0)
        assert 0 not in nbrs

    def test_self_loop_included_if_diagonal_nonzero(self):
        m = self._m()
        A = np.array([[1, 0], [0, 1]], dtype=float)  # only self-loops
        nbrs = m._neighbor_indices(A, 0)
        assert 0 in nbrs

    def test_weighted_nonzero_included(self):
        m = self._m()
        A = np.array([[0, 5, 0], [5, 0, 0.01], [0, 0.01, 0]], dtype=float)
        nbrs = m._neighbor_indices(A, 0)
        assert 1 in nbrs
        assert 2 not in nbrs

    def test_returns_ndarray(self, triangle):
        m = self._m()
        nbrs = m._neighbor_indices(triangle, 0)
        assert isinstance(nbrs, np.ndarray)

    def test_all_connected_all_neighbors(self, triangle):
        m = self._m()
        nbrs = m._neighbor_indices(triangle, 0)
        assert set(nbrs) == {1, 2}


# ─────────────────────────────────────────────────────────────────────────────
# 4. _weighted_label_vote
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedLabelVote:
    def _m(self):
        return LPCD(random_state=0)

    def test_isolated_node_keeps_label(self, isolated_node_graph):
        m = self._m()
        rng = np.random.default_rng(0)
        labels = np.array([0, 0, 2])
        vote = m._weighted_label_vote(isolated_node_graph, labels, 2, rng)
        assert vote == 2  # isolated → keeps own label

    def test_unanimous_vote_returned(self):
        m = self._m()
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        rng = np.random.default_rng(0)
        labels = np.array([99, 7, 7])
        vote = m._weighted_label_vote(A, labels, 0, rng)
        assert vote == 7  # both neighbours label=7

    def test_majority_wins(self):
        m = self._m()
        # Node 0 connected to nodes 1,2,3: labels [7,7,9] → 7 wins
        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = A[0, 2] = A[0, 3] = 1
        A[1, 0] = A[2, 0] = A[3, 0] = 1
        rng = np.random.default_rng(0)
        labels = np.array([0, 7, 7, 9])
        vote = m._weighted_label_vote(A, labels, 0, rng)
        assert vote == 7

    def test_weight_breaks_count_tie(self):
        """One neighbour with high weight beats two with low weight."""
        m = self._m()
        A = np.array([
            [0,  100, 1,  1],
            [100, 0,  0,  0],
            [1,   0,  0,  0],
            [1,   0,  0,  0],
        ], dtype=float)
        rng = np.random.default_rng(0)
        labels = np.array([0, 5, 9, 9])  # node 1→label 5 (weight=100) vs nodes 2,3→label 9 (weight=1 each)
        vote = m._weighted_label_vote(A, labels, 0, rng)
        assert vote == 5  # weighted winner

    def test_tie_returns_one_of_tied_labels(self):
        m = self._m()
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        rng = np.random.default_rng(42)
        # Nodes 1 and 2 have different labels with equal weight → tie
        labels = np.array([0, 7, 9])
        vote = m._weighted_label_vote(A, labels, 0, rng)
        assert vote in {7, 9}


# ─────────────────────────────────────────────────────────────────────────────
# 5. _compress_labels
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressLabels:
    def _m(self):
        return LPCD()

    def test_already_consecutive_unchanged(self):
        m = self._m()
        labels = np.array([0, 0, 1, 1, 2])
        result = m._compress_labels(labels)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2])

    def test_gaps_removed(self):
        m = self._m()
        labels = np.array([4, 4, 9, 9, 9])
        result = m._compress_labels(labels)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 1])

    def test_single_label(self):
        m = self._m()
        labels = np.array([5, 5, 5])
        result = m._compress_labels(labels)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_all_different(self):
        m = self._m()
        labels = np.array([10, 20, 30])
        result = m._compress_labels(labels)
        np.testing.assert_array_equal(sorted(result), [0, 1, 2])

    def test_returns_ndarray(self):
        m = self._m()
        result = m._compress_labels(np.array([3, 3, 7]))
        assert isinstance(result, np.ndarray)

    def test_output_starts_at_zero(self):
        m = self._m()
        result = m._compress_labels(np.array([99, 42, 99]))
        assert result.min() == 0

    def test_output_consecutive(self):
        m = self._m()
        labels = np.array([100, 200, 100, 300])
        result = m._compress_labels(labels)
        unique = sorted(np.unique(result))
        assert unique == list(range(len(unique)))

    def test_preserves_grouping(self):
        m = self._m()
        labels = np.array([7, 7, 3, 3, 7])
        result = m._compress_labels(labels)
        assert result[0] == result[1] == result[4]
        assert result[2] == result[3]
        assert result[0] != result[2]


# ─────────────────────────────────────────────────────────────────────────────
# 6. fit() — attribute population
# ─────────────────────────────────────────────────────────────────────────────

class TestFitAttributes:
    def test_fit_returns_self(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        assert m.fit(two_cliques) is m

    def test_labels_set_after_fit(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.labels_ is not None

    def test_labels_shape(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.labels_.shape == (6,)

    def test_labels_dtype_int_or_object(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        # Labels are integers (may be Python int or numpy int)
        assert np.issubdtype(m.labels_.dtype, np.integer) or m.labels_.dtype == object

    def test_n_iter_at_least_1(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.n_iter_ >= 1

    def test_n_iter_at_most_max_iter(self, two_cliques):
        max_iter = 10
        m = LPCD(max_iter=max_iter, random_state=0)
        m.fit(two_cliques)
        assert m.n_iter_ <= max_iter

    def test_n_communities_at_least_1(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.n_communities_ >= 1

    def test_n_communities_equals_unique_labels(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.n_communities_ == len(np.unique(m.labels_))

    def test_labels_compressed_from_zero(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        assert m.labels_.min() == 0

    def test_labels_consecutive(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(two_cliques)
        unique = sorted(np.unique(m.labels_))
        assert unique == list(range(len(unique)))


# ─────────────────────────────────────────────────────────────────────────────
# 7. fit_predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestFitPredict:
    def test_returns_ndarray(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        result = m.fit_predict(two_cliques)
        assert isinstance(result, np.ndarray)

    def test_matches_labels_(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        result = m.fit_predict(two_cliques)
        np.testing.assert_array_equal(result, m.labels_)

    def test_shape(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        result = m.fit_predict(two_cliques)
        assert result.shape == (6,)

    def test_populates_n_communities(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit_predict(two_cliques)
        assert m.n_communities_ >= 1

    def test_populates_n_iter(self, two_cliques):
        m = LPCD(max_iter=50, random_state=0)
        m.fit_predict(two_cliques)
        assert m.n_iter_ >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Community detection correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCommunityDetection:
    def test_two_disjoint_cliques_two_communities(self, two_cliques):
        m = LPCD(max_iter=100, random_state=42)
        m.fit(two_cliques)
        assert m.n_communities_ == 2

    def test_two_cliques_same_label_within_group(self, two_cliques):
        m = LPCD(max_iter=100, random_state=42)
        m.fit(two_cliques)
        assert m.labels_[0] == m.labels_[1] == m.labels_[2]
        assert m.labels_[3] == m.labels_[4] == m.labels_[5]

    def test_two_cliques_different_labels_across_groups(self, two_cliques):
        m = LPCD(max_iter=100, random_state=42)
        m.fit(two_cliques)
        assert m.labels_[0] != m.labels_[3]

    def test_full_clique_one_community(self, triangle):
        m = LPCD(max_iter=50, random_state=42)
        m.fit(triangle)
        assert m.n_communities_ == 1

    def test_full_clique_all_same_label(self, triangle):
        m = LPCD(max_iter=50, random_state=42)
        m.fit(triangle)
        assert m.labels_[0] == m.labels_[1] == m.labels_[2]

    def test_isolated_node_own_community(self, isolated_node_graph):
        m = LPCD(max_iter=50, random_state=0)
        m.fit(isolated_node_graph)
        # Node 2 is isolated → has its own label
        assert m.labels_[2] not in [m.labels_[0], m.labels_[1]]

    def test_single_node_one_community(self):
        A = np.array([[0.0]])
        m = LPCD(max_iter=5, random_state=0)
        m.fit(A)
        assert m.n_communities_ == 1
        assert m.labels_[0] == 0

    def test_all_isolated_each_own_community(self):
        A = np.zeros((4, 4), dtype=float)
        m = LPCD(max_iter=20, random_state=0)
        m.fit(A)
        assert m.n_communities_ == 4
        # All labels distinct
        assert len(np.unique(m.labels_)) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 9. Weighted adjacency matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedGraph:
    def test_strong_intra_edges_dominate(self, weighted_two_communities):
        """Strong intra-community edges (weight=10) should dominate the weak bridge."""
        m = LPCD(max_iter=100, random_state=42)
        m.fit(weighted_two_communities)
        # Nodes 0,1,2 should share a label; nodes 3,4,5 should share another
        assert m.labels_[0] == m.labels_[1] == m.labels_[2]
        assert m.labels_[3] == m.labels_[4] == m.labels_[5]
        assert m.labels_[0] != m.labels_[3]

    def test_two_communities_found_weighted(self, weighted_two_communities):
        m = LPCD(max_iter=100, random_state=42)
        m.fit(weighted_two_communities)
        assert m.n_communities_ == 2

    def test_strong_single_edge_beats_two_weak(self):
        """
        Node 0: neighbour 1 (weight=100), neighbours 2 and 3 (weight=1 each).
        After many iterations, node 0 should adopt label of node 1's community.
        """
        A = np.array([
            [0,  100, 1,   1  ],
            [100, 0,  0,   0  ],
            [1,   0,  0,   0  ],
            [1,   0,  0,   0  ],
        ], dtype=float)
        m = LPCD(max_iter=50, random_state=0)
        m.fit(A)
        assert m.labels_[0] == m.labels_[1]

    def test_uniform_weights_same_as_unweighted(self, two_cliques):
        """Scaling all weights by a constant should not change communities."""
        A_scaled = two_cliques * 5
        m1 = LPCD(max_iter=100, random_state=42)
        m2 = LPCD(max_iter=100, random_state=42)
        m1.fit(two_cliques)
        m2.fit(A_scaled)
        assert m1.n_communities_ == m2.n_communities_


# ─────────────────────────────────────────────────────────────────────────────
# 10. Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_converges_before_max_iter_on_simple_graph(self, two_cliques):
        """Two disconnected cliques converge quickly — fewer than max_iter rounds."""
        m = LPCD(max_iter=1000, random_state=42)
        m.fit(two_cliques)
        assert m.n_iter_ < 1000

    def test_max_iter_1_runs_exactly_once(self):
        A = np.ones((4, 4), dtype=float) - np.eye(4)
        m = LPCD(max_iter=1, random_state=0)
        m.fit(A)
        assert m.n_iter_ == 1

    def test_n_iter_respects_max_iter_upper_bound(self, path_graph):
        max_iter = 5
        m = LPCD(max_iter=max_iter, random_state=0)
        m.fit(path_graph)
        assert m.n_iter_ <= max_iter


# ─────────────────────────────────────────────────────────────────────────────
# 11. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_labels(self, two_cliques):
        m1 = LPCD(max_iter=50, random_state=7)
        m2 = LPCD(max_iter=50, random_state=7)
        m1.fit(two_cliques)
        m2.fit(two_cliques)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_same_seed_same_n_communities(self, weighted_two_communities):
        m1 = LPCD(max_iter=100, random_state=13)
        m2 = LPCD(max_iter=100, random_state=13)
        m1.fit(weighted_two_communities)
        m2.fit(weighted_two_communities)
        assert m1.n_communities_ == m2.n_communities_

    def test_same_seed_same_n_iter(self, two_cliques):
        m1 = LPCD(max_iter=50, random_state=99)
        m2 = LPCD(max_iter=50, random_state=99)
        m1.fit(two_cliques)
        m2.fit(two_cliques)
        assert m1.n_iter_ == m2.n_iter_

    def test_fit_predict_consistent_with_fit(self, two_cliques):
        m1 = LPCD(max_iter=50, random_state=5)
        m2 = LPCD(max_iter=50, random_state=5)
        m1.fit(two_cliques)
        fp = m2.fit_predict(two_cliques)
        np.testing.assert_array_equal(fp, m1.labels_)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Integration — structured community graph
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def _build_community_graph(self, community_sizes, intra=5.0, inter=0.1, seed=0):
        """
        Build a block-structured adjacency matrix where nodes within each
        community are strongly connected and inter-community connections are weak.
        """
        n = sum(community_sizes)
        A = np.zeros((n, n), dtype=float)
        rng = np.random.default_rng(seed)
        starts = np.cumsum([0] + community_sizes)
        for k, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
            for i in range(s, e):
                for j in range(s, e):
                    if i != j:
                        A[i, j] = intra
        # Sparse inter-community links
        for _ in range(3):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                A[i, j] = inter
                A[j, i] = inter
        return A

    def test_three_community_graph(self):
        """Block model with 3 communities should detect exactly 3."""
        A = self._build_community_graph([8, 8, 8], intra=10.0, inter=0.01, seed=1)
        m = LPCD(max_iter=200, random_state=42)
        m.fit(A)
        assert m.n_communities_ == 3

    def test_intra_community_same_label(self):
        """All nodes within a tightly connected community should share a label."""
        A = self._build_community_graph([6, 6], intra=20.0, inter=0.001, seed=2)
        m = LPCD(max_iter=200, random_state=0)
        m.fit(A)
        labels_c0 = m.labels_[:6]
        labels_c1 = m.labels_[6:]
        assert len(np.unique(labels_c0)) == 1
        assert len(np.unique(labels_c1)) == 1

    def test_labels_shape_large_graph(self):
        A = self._build_community_graph([10, 10, 10, 10], intra=5.0, inter=0.05)
        m = LPCD(max_iter=100, random_state=0)
        m.fit(A)
        assert m.labels_.shape == (40,)

    def test_n_communities_compressed_consecutive(self):
        A = self._build_community_graph([5, 5, 5], intra=8.0, inter=0.01)
        m = LPCD(max_iter=150, random_state=7)
        m.fit(A)
        unique = sorted(np.unique(m.labels_))
        assert unique == list(range(len(unique)))

    def test_path_graph_multiple_communities(self, path_graph):
        """A linear chain with only local connections may fragment into communities."""
        m = LPCD(max_iter=100, random_state=0)
        m.fit(path_graph)
        # There should be at least 1 community
        assert m.n_communities_ >= 1
        # Labels should be valid compressed integers
        unique = sorted(np.unique(m.labels_))
        assert unique == list(range(len(unique)))
