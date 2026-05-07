"""
label_propagation_community_detection.py

A from-scratch implementation of Label Propagation for community detection
using NumPy.

Label propagation is an unsupervised graph algorithm used to find communities
inside a network. A community is a group of nodes that are more closely
connected to each other than to the rest of the graph.

The basic idea is simple:

1. Give every node its own unique label.
2. Visit each node and look at the labels of its neighbors.
3. Change the node's label to the label most common among its neighbors.
4. Repeat this process until labels stop changing or the maximum number of
   iterations is reached.

This implementation supports weighted adjacency matrices. That means stronger
edges can have more influence during the label vote.

This file provides:

- label_propagation_community_detection:
    A simple label propagation class for graph community detection.

- fit(X):
    Runs label propagation on an adjacency matrix.

- fit_predict(X):
    Fits the model and returns the final community labels.

The model stores:

- labels_:
    The final community label assigned to each node.

- n_iter_:
    The number of update rounds completed.

- n_communities_:
    The number of communities found.

This implementation is intended for educational use and does not rely on
networkx or scikit-learn.
"""

import numpy as np


class label_propagation_community_detection:
    def __init__(self, max_iter=100, random_state=None):
        """
        Create a label propagation community detection model.

        Parameters
        ----------
        max_iter : int
            Maximum number of full update rounds through the graph.

        random_state : int or None
            Controls random update order and tie-breaking so results can be
            reproduced.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_ = None
        self.n_iter_ = 0
        self.n_communities_ = 0

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------

    def _prepare_adjacency_matrix(self, X):
        """
        Convert input into a valid square adjacency matrix.

        Parameters
        ----------
        X : array-like of shape (n_nodes, n_nodes)
            Adjacency matrix representing a graph.

            X[i, j] is the edge weight between node i and node j.
            A value of 0 means there is no edge.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D adjacency matrix.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one node.")

        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be square with shape (n_nodes, n_nodes).")

        if np.any(X < 0):
            raise ValueError("Adjacency matrix edge weights must be nonnegative.")

        return X

    # ---------------------------------------------------------
    # Neighbor voting
    # ---------------------------------------------------------

    def _neighbor_indices(self, adjacency_matrix, node_index):
        """
        Return the indices of all nodes connected to a given node.
        """
        return np.where(adjacency_matrix[node_index] > 0)[0]

    def _weighted_label_vote(self, adjacency_matrix, labels, node_index, rng):
        """
        Choose a new label for one node based on its neighbors.

        Each neighbor votes using its current label. If the graph is weighted,
        the neighbor's edge weight determines the strength of the vote.

        If there is a tie, one of the tied labels is chosen randomly.
        """
        neighbors = self._neighbor_indices(adjacency_matrix, node_index)

        # Isolated nodes keep their current label.
        if len(neighbors) == 0:
            return labels[node_index]

        vote_totals = {}

        for neighbor in neighbors:
            neighbor_label = labels[neighbor]
            edge_weight = adjacency_matrix[node_index, neighbor]

            if neighbor_label not in vote_totals:
                vote_totals[neighbor_label] = 0.0

            vote_totals[neighbor_label] += edge_weight

        best_score = max(vote_totals.values())

        best_labels = [
            label
            for label, score in vote_totals.items()
            if score == best_score
        ]

        if len(best_labels) == 1:
            return best_labels[0]

        return rng.choice(best_labels)

    # ---------------------------------------------------------
    # Label cleanup
    # ---------------------------------------------------------

    def _compress_labels(self, labels):
        """
        Convert final labels into clean consecutive labels.

        Example:
        [4, 4, 9, 9, 9] becomes [0, 0, 1, 1, 1].
        """
        unique_labels = np.unique(labels)

        label_map = {
            old_label: new_label
            for new_label, old_label in enumerate(unique_labels)
        }

        compressed = np.array([
            label_map[label]
            for label in labels
        ])

        return compressed

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------

    def fit(self, X):
        """
        Run label propagation on a graph.

        Parameters
        ----------
        X : array-like of shape (n_nodes, n_nodes)
            A square adjacency matrix.

            Each row and column represents a node.
            X[i, j] gives the edge weight between node i and node j.
            If X[i, j] is 0, there is no edge from node i to node j.

        Returns
        -------
        self
            The fitted community detection model.
        """
        adjacency_matrix = self._prepare_adjacency_matrix(X)

        n_nodes = adjacency_matrix.shape[0]
        rng = np.random.default_rng(self.random_state)

        # At the beginning, every node starts in its own community.
        labels = np.arange(n_nodes)

        for iteration in range(self.max_iter):
            old_labels = labels.copy()

            # Random update order helps avoid repeated update cycles.
            update_order = rng.permutation(n_nodes)

            for node_index in update_order:
                labels[node_index] = self._weighted_label_vote(
                    adjacency_matrix,
                    labels,
                    node_index,
                    rng
                )

            self.n_iter_ = iteration + 1

            # Stop early if a full pass causes no label changes.
            if np.array_equal(labels, old_labels):
                break

        self.labels_ = self._compress_labels(labels)
        self.n_communities_ = len(np.unique(self.labels_))

        return self

    def fit_predict(self, X):
        """
        Fit the model and return the final community labels.

        Parameters
        ----------
        X : array-like of shape (n_nodes, n_nodes)
            Square adjacency matrix for the graph.

        Returns
        -------
        np.ndarray
            Community label for each node.
        """
        self.fit(X)

        return self.labels_