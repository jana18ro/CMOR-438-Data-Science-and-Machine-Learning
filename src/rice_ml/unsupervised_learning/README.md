# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\unsupervised_learning\README.md

```markdown
# unsupervised_learning

From-scratch implementations of unsupervised learning algorithms for clustering,
dimensionality reduction, and graph-based community detection.

---

## Algorithms

### Clustering

#### `kmeans.py` — `KMeans`
Centroid-based clustering implementing Lloyd's algorithm.
Convergence is determined by centroid movement falling below a tolerance `tol`.
- **Parameters:** `n_clusters`, `max_iter`, `tol`, `random_state`
- **Attributes:** `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`
- **Methods:** `fit`, `predict`, `fit_predict`, `score` (negative inertia)
- **Use cases:** Customer segmentation, image quantisation, document clustering.
- **Elbow method:** Plot `inertia_` vs. `n_clusters` to select optimal k.

#### `dbscan.py` — `DBSCAN`
Density-Based Spatial Clustering of Applications with Noise.
Identifies arbitrarily shaped clusters and labels outliers as noise (`-1`).
Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance.
- **Parameters:** `eps` (neighbourhood radius), `min_samples`
- **Attributes:** `labels_`, `core_sample_indices_`, `n_clusters_`
- **Methods:** `fit`, `fit_predict`
- **Use cases:** Geospatial data, anomaly detection, non-convex cluster shapes.

---

### Dimensionality Reduction

#### `pca.py` — `PCA`
Principal Component Analysis via eigen-decomposition of the covariance matrix.
Finds the directions (principal components) of maximum variance in the data.
- **Parameters:** `n_components`
- **Attributes:** `components_`, `explained_variance_`, `explained_variance_ratio_`,
  `mean_`
- **Methods:** `fit`, `transform`, `fit_transform`, `inverse_transform`
- **Use cases:** Visualisation (4D → 2D), noise reduction, preprocessing for other models.

#### `svd.py` — `SVD`
Singular Value Decomposition: factorises matrix A into U Σ Vᵀ.
Supports truncated (rank-k) approximation for dimensionality reduction and
image compression demonstrations.
- **Parameters:** `n_components`
- **Attributes:** `U_`, `sigma_`, `Vt_`, `explained_variance_ratio_`
- **Methods:** `fit`, `transform`, `fit_transform`, `reconstruct`

---

### Graph Analysis

#### `label_propagation_community_detection.py` — `LabelPropagation`
Community detection via iterative neighbourhood label spreading.
At each iteration every node adopts the most frequent label among its neighbours
until the labelling stabilises (converges).
- **Parameters:** `max_iter`, `random_state`
- **Attributes:** `labels_`, `n_communities_`, `n_iter_`
- **Methods:** `fit` (accepts adjacency matrix or edge list), `fit_predict`
- **Use cases:** Social network analysis, biological network clustering,
  citation network community discovery.

---

## Example Usage

```python
from rice_ml.unsupervised_learning import KMeans, PCA
from rice_ml.processing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction before clustering
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print("Variance retained:", pca.explained_variance_ratio_.sum())

# Clustering in 2D
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_2d)
labels = kmeans.predict(X_2d)
print("Inertia:", kmeans.inertia_)
```

See the corresponding notebooks in [`examples/unsupervised_learning/`](../../../examples/unsupervised_learning/).
```

---