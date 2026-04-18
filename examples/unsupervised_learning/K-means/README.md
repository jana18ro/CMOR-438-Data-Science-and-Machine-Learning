# FILE: 2026_Data_Science_and_Machine_Learning\examples\unsupervised_learning\K-means\README.md

```markdown
# K-means/

Demonstrates K-Means clustering using `rice_ml.unsupervised_learning.KMeans`.

## Notebook

`kmeans_notebook.ipynb`

## What the Notebook Covers

- **Lloyd's Algorithm:** Centroid initialisation, assignment step, update step;
  visualised iteration-by-iteration.
- **Inertia:** Within-cluster sum of squares as a convergence criterion.
- **Choosing k:** Elbow method (inertia vs. k plot) and silhouette score comparison.
- **Cluster Visualisation:** 2D scatter plots with colour-coded cluster assignments
  and centroid markers.
- **Limitations:** Sensitivity to initialisation, assumption of convex/spherical clusters.
- **Comparison:** K-Means vs. ground-truth Iris species labels.

## Dataset

- `sklearn.datasets.make_blobs` (well-separated synthetic clusters).
- Iris dataset (natural cluster structure).

## Key imports

```python
from rice_ml.unsupervised_learning import KMeans
from rice_ml.processing import StandardScaler
from rice_ml.measures import accuracy_score
```
```

---
