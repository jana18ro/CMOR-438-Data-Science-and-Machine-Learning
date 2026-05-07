"""
pca.py

A from-scratch implementation of Principal Component Analysis using NumPy.

Principal Component Analysis, or PCA, is an unsupervised learning method used
for dimensionality reduction. It transforms a dataset with many features into a
smaller set of new features called principal components.

The principal components are chosen so that they capture as much variation in
the original data as possible. The first principal component captures the most
variation, the second captures the next most, and so on.

This file provides:

- pca:
    A simple PCA class for reducing the dimensionality of numerical datasets.

- fit(X):
    Learns the feature means, principal components, explained variance, and
    explained variance ratios from the training data.

- transform(X):
    Projects data from the original feature space into the lower-dimensional
    principal component space.

- fit_transform(X):
    Fits the PCA model and returns the transformed data.

- inverse_transform(X):
    Projects reduced data back into the original feature space.

- score(X):
    Returns the negative reconstruction error. A higher score is better because
    lower reconstruction error means the reduced representation preserved more
    information.

The model stores:

- components_:
    The principal component directions.

- explained_variance_:
    The amount of variance explained by each selected component.

- explained_variance_ratio_:
    The fraction of total variance explained by each selected component.

- singular_values_:
    Singular values from the centered data matrix.

- mean_:
    The feature means learned from the training data.

This implementation is intended for educational use and does not rely on
scikit-learn.
"""

import numpy as np


class pca:
    def __init__(self, n_components=None):
        """
        Create a PCA model.

        Parameters
        ----------
        n_components : int, float, or None
            Controls how many principal components are kept.

            If n_components is an int:
                Keep exactly that many components.

            If n_components is a float between 0 and 1:
                Keep enough components to explain at least that fraction of
                the total variance.

                Example:
                n_components=0.95 keeps enough components to explain at least
                95% of the variance.

            If n_components is None:
                Keep all possible components.
        """
        self.n_components = n_components

        self.n_components_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------

    def _prepare_X(self, X):
        """
        Convert input data into a two-dimensional numeric NumPy array.
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
    # Component selection
    # ---------------------------------------------------------

    def _choose_number_of_components(self, explained_variance_ratio, max_components):
        """
        Decide how many principal components to keep.
        """
        if self.n_components is None:
            return max_components

        if isinstance(self.n_components, int):
            if self.n_components < 1:
                raise ValueError("n_components must be at least 1.")

            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than the number of available components."
                )

            return self.n_components

        if isinstance(self.n_components, float):
            if not 0 < self.n_components < 1:
                raise ValueError(
                    "When n_components is a float, it must be between 0 and 1."
                )

            cumulative_variance = np.cumsum(explained_variance_ratio)

            return int(np.searchsorted(cumulative_variance, self.n_components) + 1)

        raise ValueError("n_components must be an int, float, or None.")

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The numerical dataset.

        Returns
        -------
        self
            The fitted PCA model.
        """
        X = self._prepare_X(X)

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")

        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Center the data by subtracting each feature's mean.
        self.mean_ = np.mean(X, axis=0)
        centered_X = X - self.mean_

        # Singular Value Decomposition:
        # centered_X = U @ S @ Vt
        #
        # Rows of Vt are the principal component directions.
        _, singular_values, Vt = np.linalg.svd(
            centered_X,
            full_matrices=False
        )

        all_explained_variance = (singular_values ** 2) / (n_samples - 1)

        total_variance = np.sum(all_explained_variance)

        if total_variance == 0:
            all_explained_variance_ratio = np.zeros_like(all_explained_variance)
        else:
            all_explained_variance_ratio = all_explained_variance / total_variance

        max_components = min(n_samples, n_features)

        selected_components = self._choose_number_of_components(
            all_explained_variance_ratio,
            max_components
        )

        self.n_components_ = selected_components
        self.components_ = Vt[:selected_components]
        self.explained_variance_ = all_explained_variance[:selected_components]
        self.explained_variance_ratio_ = all_explained_variance_ratio[:selected_components]
        self.singular_values_ = singular_values[:selected_components]

        return self

    # ---------------------------------------------------------
    # Transforming data
    # ---------------------------------------------------------

    def transform(self, X):
        """
        Project data into the principal component space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        if self.components_ is None:
            raise RuntimeError("The PCA model must be fitted before transformation.")

        X = self._prepare_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X must have the same number of features as the training data."
            )

        centered_X = X - self.mean_

        return centered_X @ self.components_.T

    def fit_transform(self, X):
        """
        Fit PCA to X and return the transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X):
        """
        Project reduced PCA data back into the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        np.ndarray
            Approximation of the original data.
        """
        if self.components_ is None:
            raise RuntimeError("The PCA model must be fitted before inverse transformation.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[1] != self.n_components_:
            raise ValueError(
                "X must have the same number of columns as the selected PCA components."
            )

        return X @ self.components_ + self.mean_

    # ---------------------------------------------------------
    # Reconstruction and scoring
    # ---------------------------------------------------------

    def reconstruction_error(self, X):
        """
        Calculate the mean squared reconstruction error.

        This measures how much information was lost when reducing the data and
        reconstructing it back into the original feature space.
        """
        X = self._prepare_X(X)

        reduced_X = self.transform(X)
        reconstructed_X = self.inverse_transform(reduced_X)

        return float(np.mean((X - reconstructed_X) ** 2))

    def score(self, X):
        """
        Return the negative reconstruction error.

        Higher is better because smaller reconstruction error means the PCA
        representation preserved more of the original data.
        """
        return -self.reconstruction_error(X)

    # ---------------------------------------------------------
    # Extra utility methods
    # ---------------------------------------------------------

    def get_covariance(self):
        """
        Estimate the covariance matrix using the selected principal components.

        Returns
        -------
        np.ndarray
            Approximate covariance matrix in the original feature space.
        """
        if self.components_ is None:
            raise RuntimeError("The PCA model must be fitted before covariance is computed.")

        return (
            self.components_.T
            @ np.diag(self.explained_variance_) # type: ignore
            @ self.components_
        )