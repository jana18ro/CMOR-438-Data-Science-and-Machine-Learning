# FILE: 2026_Data_Science_and_Machine_Learning\examples\unsupervised_learning\SVD\README.md

```markdown
# SVD/

Demonstrates Singular Value Decomposition using `rice_ml.unsupervised_learning.SVD`,
with a focus on low-rank matrix approximation and image compression.

## Notebook

`svd_notebook.ipynb`

## What the Notebook Covers

- **SVD Factorisation:** A = U Σ Vᵀ — interpretation of U (left singular vectors),
  Σ (singular values), and Vᵀ (right singular vectors).
- **Relationship to PCA:** How truncated SVD and PCA relate; when to prefer each.
- **Low-Rank Approximation:** Rank-k reconstruction at k = 1, 5, 10, 50, 100;
  visual comparison of image quality vs. compression ratio.
- **Reconstruction Error:** Frobenius norm of (A − Â_k) as a function of rank k.
- **Explained Variance:** Fraction of variance captured by the top-k singular values.
- **Latent Feature Extraction:** Applying SVD to a document-term matrix (LSA demo).

## Dataset

A sample grayscale image represented as a 2D NumPy array.

## Key imports

```python
from rice_ml.unsupervised_learning import SVD, PCA
```
```

---