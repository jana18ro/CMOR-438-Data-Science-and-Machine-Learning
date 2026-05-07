# Unsupervise Learning

# examples/unsupervised_learning/

Jupyter notebooks demonstrating the unsupervised learning algorithms from
`rice_ml.unsupervised_learning` on real and synthetic datasets.

Each notebook covers algorithm intuition, step-by-step implementation walkthrough,
hyperparameter sensitivity analysis, and result visualisation.

---

## Contents

| Folder | Algorithm | Key Concepts | Dataset |
|--------|-----------|-------------|---------|
| [`K-means/`](K-means/) | K-Means Clustering | Lloyd's algorithm, inertia, elbow method | mall customers |
| [`DBSCAN/`](DBSCAN/) | Density-Based Clustering | ε, min_samples, noise points, arbitrary shapes | `make_moons`, `make_circles` |
| [`PCA/`](PCA/) | Principal Component Analysis | Eigenvectors, explained variance, scree plot | Iris (4D → 2D) |
| [`Label_Propagation_Community_Detection/`](Label_Propagation_Community_Detection/) | Community Detection | Graph adjacency, label spreading, convergence | Synthetic |
```
