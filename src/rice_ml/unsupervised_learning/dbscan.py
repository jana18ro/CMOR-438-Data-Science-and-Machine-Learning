"""
dbscan.py

A from-scratch implementation of the DBSCAN clustering algorithm using NumPy.

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
It is an unsupervised learning algorithm that groups points together based on
density rather than requiring the user to choose the number of clusters ahead
of time.

This implementation identifies three kinds of points:

- Core points:
    Points that have at least `min_samples` points within distance `eps`,
    including themselves.

- Border points:
    Points that do not have enough nearby points to start their own cluster,
    but are close enough to a core point to be included in that core point's
    cluster.

- Noise points:
    Points that do not belong to any cluster. These are labeled as -1.

The model follows a simple sklearn-like structure with:

- fit(X):
    Learns cluster assignments from the input data.

- fit_predict(X):
    Fits the model and immediately returns the cluster labels.

Cluster labels are stored in the `labels_` attribute after fitting.

Example
-------
>>> import numpy as np
>>> from dbscan import dbscan
>>>
>>> X = np.array([
...     [0.0, 0.0],
...     [0.1, 0.1],
...     [0.2, 0.0],
...     [5.0, 5.0],
...     [5.1, 5.1],
...     [9.0, 9.0]
... ])
>>>
>>> model = dbscan(eps=0.35, min_samples=2)
>>> model.fit(X)
>>> model.labels_
array([ 0,  0,  0,  1,  1, -1])
"""

import numpy as np
from collections import deque


class dbscan:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Create a DBSCAN clustering model.

        Parameters
        ----------
        eps : float
            The maximum distance two points can be from each other and still be
            considered neighbors.

        min_samples : int
            The minimum number of points required in a point's neighborhood for
            that point to be considered a core point.

            The neighborhood includes the point itself.
        """
        if eps <= 0:
            raise ValueError("eps must be positive.")

        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")

        self.eps = eps
        self.min_samples = min_samples

        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------

    def _prepare_X(self, X):
        """
        Convert input data into a two-dimensional NumPy array.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        return X

    # ---------------------------------------------------------
    # Distance and neighborhood search
    # ---------------------------------------------------------

    def _euclidean_distances_from_point(self, X, point_index):
        """
        Calculate the Euclidean distance from one point to every point in X.
        """
        differences = X - X[point_index]
        squared_distances = np.sum(differences ** 2, axis=1)

        return np.sqrt(squared_distances)

    def _find_neighbors(self, X, point_index):
        """
        Return the indices of all points within eps distance of one point.
        """
        distances = self._euclidean_distances_from_point(X, point_index)

        return np.where(distances <= self.eps)[0]

    # ---------------------------------------------------------
    # Cluster expansion
    # ---------------------------------------------------------

    def _expand_cluster(
        self,
        X,
        labels,
        visited,
        start_index,
        start_neighbors,
        cluster_id,
        core_points
    ):
        """
        Grow a cluster outward from a starting core point.

        DBSCAN expands a cluster by repeatedly checking whether neighboring
        points are also core points. If they are, their neighbors are added to
        the cluster search.
        """
        labels[start_index] = cluster_id

        points_to_check = deque(start_neighbors.tolist())

        while points_to_check:
            current_index = points_to_check.popleft()

            if not visited[current_index]:
                visited[current_index] = True

                current_neighbors = self._find_neighbors(X, current_index)

                if len(current_neighbors) >= self.min_samples:
                    core_points.append(current_index)

                    for neighbor_index in current_neighbors:
                        if neighbor_index not in points_to_check:
                            points_to_check.append(neighbor_index)

            if labels[current_index] == -1:
                labels[current_index] = cluster_id

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------

    def fit(self, X):
        """
        Fit DBSCAN to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        self
            The fitted DBSCAN model.
        """
        X = self._prepare_X(X)

        n_samples = X.shape[0]

        # Every point begins as noise.
        labels = np.full(n_samples, -1, dtype=int)

        # Tracks whether we have already examined a point.
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0
        core_points = []

        for point_index in range(n_samples):
            if visited[point_index]:
                continue

            visited[point_index] = True

            neighbors = self._find_neighbors(X, point_index)

            # Not enough nearby points means this point cannot start a cluster.
            if len(neighbors) < self.min_samples:
                labels[point_index] = -1

            else:
                core_points.append(point_index)

                self._expand_cluster(
                    X=X,
                    labels=labels,
                    visited=visited,
                    start_index=point_index,
                    start_neighbors=neighbors,
                    cluster_id=cluster_id,
                    core_points=core_points
                )

                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(sorted(set(core_points)), dtype=int)
        self.n_clusters_ = cluster_id

        return self

    def fit_predict(self, X):
        """
        Fit DBSCAN to X and return the cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        np.ndarray
            Cluster labels for each sample.

            Noise points are labeled as -1.
        """
        self.fit(X)

        return self.labels_