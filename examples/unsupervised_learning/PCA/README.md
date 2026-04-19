# Principal Component Analysis

```markdown
# PCA/

Demonstrates Principal Component Analysis using `rice_ml.unsupervised_learning.PCA`.

## Notebook

`pca_notebook.ipynb`

## What the Notebook Covers

- **Covariance Matrix:** Computation and interpretation.
- **Eigenvectors and Eigenvalues:** Principal component directions and their magnitudes.
- **Explained Variance:** Individual and cumulative explained variance ratio;
  scree plot for choosing the number of components.
- **2D Projection:** Reducing Iris from 4D to 2D; visualising class separation
  in principal component space.
- **PCA as Preprocessing:** Fitting PCA on training data only; applying the same
  transformation to test data without refitting.
- **Reconstruction:** Projecting back to original space and computing reconstruction error.
- **Comparison:** Classification accuracy with vs. without PCA preprocessing.

## Dataset

Iris dataset (4 features → 2 principal components).

## Key imports

```python
from rice_ml.unsupervised_learning import PCA
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.supervised_learning import LogisticRegression
from rice_ml.measures import accuracy_score
```
```

---
