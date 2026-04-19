# DBSCAN Clustering

```markdown
# DBSCAN/

Demonstrates density-based clustering using `rice_ml.unsupervised_learning.DBSCAN`.

## Notebook

`dbscan_notebook.ipynb`

## What the Notebook Covers

- **Core, Border, and Noise Points:** Definition and identification with diagrams.
- **Parameter Sensitivity:** How `eps` (neighbourhood radius) and `min_samples`
  together define cluster density; grid search visualisation.
- **Arbitrary Cluster Shapes:** Side-by-side comparison of DBSCAN vs. K-Means on
  crescent, circular, and spiral datasets — where K-Means fails and DBSCAN succeeds.
- **Noise Detection:** Identifying and visualising outlier points (label = -1).
- **k-distance Plot:** Heuristic for selecting a suitable `eps` value.

## Dataset

- `sklearn.datasets.make_moons` (two interleaved crescents).
- `sklearn.datasets.make_circles` (concentric circles with noise).

## Key imports

```python
from rice_ml.unsupervised_learning import DBSCAN, KMeans
from rice_ml.processing import StandardScaler
```
```

---
