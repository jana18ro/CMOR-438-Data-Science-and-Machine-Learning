"""
rice_ml.unsupervised_learning
==============================
Unsupervised learning algorithms implemented from scratch using NumPy.

Available Estimators
--------------------
kmeans / KMeans
    Centroid-based clustering implementing Lloyd's algorithm.

dbscan / DBSCAN
    Density-Based Spatial Clustering of Applications with Noise.

pca / PCA
    Dimensionality reduction via SVD of the covariance matrix.

label_propagation_community_detection / LabelPropagation
    Graph-based community detection via iterative label spreading.
"""

from .kmeans import kmeans
from .dbscan import dbscan
from .pca import pca
from .label_propagation_community_detection import label_propagation_community_detection

# Uppercase aliases for convenience / notebooks
KMeans = kmeans
DBSCAN = dbscan
PCA = pca
LabelPropagation = label_propagation_community_detection

__all__ = [
    "kmeans",
    "KMeans",
    "dbscan",
    "DBSCAN",
    "pca",
    "PCA",
    "label_propagation_community_detection",
    "LabelPropagation",
]
