# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\unsupervised_learning\__init__.py


"""
rice_ml.unsupervised_learning
==============================
Unsupervised learning algorithms implemented from scratch using NumPy.

This subpackage provides estimators for clustering, dimensionality reduction,
and graph-based community detection. The interface mirrors the supervised
learning subpackage where applicable:

    fit(X)          -> learns structure from unlabelled data
    predict(X)      -> returns cluster labels or transformed features
    fit_transform(X)-> fits and returns the transformed representation

Available Estimators
--------------------
KMeans
    Centroid-based clustering implementing Lloyd's algorithm.
    Convergence is tracked via inertia (within-cluster sum of squares).
    Supports the elbow method and silhouette score for choosing k.
    Key parameters: ``n_clusters``, ``max_iter``, ``tol``, ``random_state``

DBSCAN
    Density-Based Spatial Clustering of Applications with Noise.
    Identifies arbitrarily shaped clusters and labels outliers as noise (-1).
    Key parameters: ``eps`` (neighbourhood radius), ``min_samples``

PCA (Principal Component Analysis)
    Dimensionality reduction via eigen-decomposition of the covariance matrix.
    Returns transformed coordinates and explained variance ratios.
    Key parameters: ``n_components``
    Attributes: ``components_``, ``explained_variance_ratio_``

SVD (Singular Value Decomposition)
    Matrix factorisation A = U Σ Vᵀ for dimensionality reduction and
    latent feature extraction. Supports truncated (rank-k) approximation,
    useful for image compression demonstrations.
    Key parameters: ``n_components``

LabelPropagation
    Graph-based community detection via iterative neighbourhood label
    spreading. At each step every node adopts the most frequent label
    among its neighbours until convergence.
    Input: adjacency matrix or edge list.

Examples
--------
>>> from rice_ml.unsupervised_learning import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
>>> model = KMeans(n_clusters=2, random_state=42)
>>> model.fit(X)
>>> model.predict(X)
array([0, 0, 0, 1, 1, 1])

>>> from rice_ml.unsupervised_learning import PCA
>>> pca = PCA(n_components=2)
>>> X_reduced = pca.fit_transform(X_high_dim)
>>> print(pca.explained_variance_ratio_)
"""

from .kmeans import KMeans
from .dbscan import DBSCAN
from .pca import PCA
from .svd import SVD
from .label_propagation_community_detection import LabelPropagation

__all__ = [
    "KMeans",
    "DBSCAN",
    "PCA",
    "SVD",
    "LabelPropagation",
]
