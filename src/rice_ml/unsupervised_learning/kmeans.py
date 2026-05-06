"""
Kmeans.py

A from-scratch implementation of the K-Means clustering algorithm using NumPy.

K-Means is an unsupervised learning algorithm that separates numerical data
into a chosen number of groups, called clusters. The algorithm works by finding
cluster centers, called centroids, and assigning each data point to the nearest
centroid.

This implementation follows the standard K-Means process:

1. Choose K initial centroids from the dataset.
2. Assign each data point to the closest centroid.
3. Recalculate each centroid as the average of the points assigned to it.
4. Repeat this process until the centroids stop moving significantly or the
   maximum number of iterations is reached.

This file provides:

- kmeans:
    A simple K-Means clustering class.

- fit(X):
    Learns cluster centroids from the data.

- predict(X):
    Assigns new data points to the nearest learned centroid.

- fit_predict(X):
    Fits the model and returns cluster labels.

- score(X):
    Returns the negative inertia, similar to the scoring style used by
    clustering models in scikit-learn.

The model stores:

- centroids:
    The final cluster centers.

- labels_:
    The cluster label assigned to each training point.

- inertia_:
    The sum of squared distances between each point and its assigned centroid.

This implementation is intended for educational use and does not rely on
scikit-learn.
"""

import numpy as np


class kmeans:
    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        tol=1e-4,
        random_state=42
    ):
        """
        Create a K-Means clustering model.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to create.

        max_iter : int
            The maximum number of times the algorithm will update cluster
            assignments and centroids.

        tol : float
            The convergence tolerance.

            If the centroids move less than this amount between iterations,
            the algorithm stops early.

        random_state : int or None
            Controls random centroid initialization so results can be
            reproduced.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        if tol < 0:
            raise ValueError("tol must be nonnegative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

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

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "n_clusters cannot be greater than the number of samples."
            )

        return X

    # ---------------------------------------------------------
    # Distance calculations
    # ---------------------------------------------------------

    def _distance_to_centroids(self, X):
        """
        Calculate the distance from every point to every centroid.

        Returns
        -------
        np.ndarray
            A distance matrix with shape:

            (n_samples, n_clusters)

            Each row represents one data point.
            Each column represents one centroid.
        """
        differences = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        squared_distances = np.sum(differences ** 2, axis=2)

        return np.sqrt(squared_distances)

    # ---------------------------------------------------------
    # Cluster assignment
    # ---------------------------------------------------------

    def _assign_labels(self, X):
        """
        Assign each point to the nearest centroid.
        """
        distances = self._distance_to_centroids(X)

        return np.argmin(distances, axis=1)

    # ---------------------------------------------------------
    # Centroid initialization
    # ---------------------------------------------------------

    def _initialize_centroids(self, X):
        """
        Randomly choose initial centroids from the dataset.
        """
        rng = np.random.default_rng(self.random_state)

        chosen_indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False
        )

        return X[chosen_indices].copy()

    # ---------------------------------------------------------
    # Centroid updating
    # ---------------------------------------------------------

    def _update_centroids(self, X, labels):
        """
        Recalculate centroid positions based on current cluster assignments.
        """
        new_centroids = []

        rng = np.random.default_rng(self.random_state)

        for cluster_id in range(self.n_clusters):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                # If a cluster becomes empty, choose a replacement point.
                replacement_index = rng.integers(0, X.shape[0])
                new_centroids.append(X[replacement_index])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))

        return np.array(new_centroids)

    # ---------------------------------------------------------
    # Inertia
    # ---------------------------------------------------------

    def _compute_inertia(self, X, labels):
        """
        Calculate inertia.

        Inertia is the sum of squared distances from each point to its assigned
        centroid. Smaller inertia usually means tighter clusters.
        """
        total = 0.0

        for cluster_id in range(self.n_clusters):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                continue

            differences = cluster_points - self.centroids[cluster_id]
            squared_distances = np.sum(differences ** 2, axis=1)

            total += np.sum(squared_distances)

        return float(total)

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The numerical dataset to cluster.

        Returns
        -------
        self
            The fitted K-Means model.
        """
        X = self._prepare_X(X)

        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)

            centroid_shift = np.linalg.norm(new_centroids - self.centroids)

            self.centroids = new_centroids
            self.n_iter_ = iteration + 1

            if centroid_shift <= self.tol:
                break

        self.labels_ = self._assign_labels(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)

        return self

    def fit_predict(self, X):
        """
        Fit K-Means to X and return the cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The numerical dataset to cluster.

        Returns
        -------
        np.ndarray
            Cluster label for each sample.
        """
        self.fit(X)

        return self.labels_

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self, X):
        """
        Assign new samples to the nearest learned centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data points.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        if self.centroids is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                "X must have the same number of features as the training data."
            )

        return self._assign_labels(X)

    def score(self, X):
        """
        Return the negative inertia.

        This follows the common convention that higher scores are better.
        Since lower inertia is better, the score is returned as negative
        inertia.
        """
        if self.centroids is None:
            raise RuntimeError("The model must be fitted before scoring.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        labels = self.predict(X)

        return -self._compute_inertia(X, labels)