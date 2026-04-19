# K-Nearest Neighbors

```markdown
# KNN/

Demonstrates K-Nearest Neighbours classification and regression using
`rice_ml.supervised_learning.KNNClassifier` and `KNNRegressor`.

## Notebook

`knn_notebook.ipynb`

## What the Notebook Covers

- Distance metrics: Euclidean vs. Manhattan — effect on decision boundaries.
- Choosing k: accuracy vs. k plots illustrating the bias-variance trade-off
  (low k = high variance, high k = high bias).
- Decision boundary visualisation for KNN Classifier in 2D.
- KNN Regressor: prediction curve vs. true function for different k values.
- Evaluation: accuracy (classification), MSE and R² (regression).

## Dataset

- **Classification:** Iris dataset (4 features, 3 classes).
- **Regression:** Synthetic sinusoidal data with noise.

## Key imports

```python
from rice_ml.supervised_learning import KNNClassifier, KNNRegressor
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, mean_squared_error, r2_score
```
```

---