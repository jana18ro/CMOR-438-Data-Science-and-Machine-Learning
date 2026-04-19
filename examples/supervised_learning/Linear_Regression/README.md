# Linear Regression

```markdown
# Linear_Regression/

Demonstrates ordinary least squares, Ridge, and Lasso regression using
`rice_ml.supervised_learning.LinearRegression`, `RidgeRegression`, and `LassoRegression`.

## Notebook

`linear_regression_notebook.ipynb`

## What the Notebook Covers

- **OLS Linear Regression:** Gradient descent solution; visualising the loss curve,
  fitted line, and residuals.
- **Ridge Regression (L2):** Effect of the regularisation parameter `alpha` on
  coefficient magnitude; bias-variance trade-off.
- **Lasso Regression (L1):** Feature selection behaviour; coefficient paths as `alpha`
  varies; sparsity in the solution.
- **Model Comparison:** MSE, RMSE, and R² across OLS, Ridge, and Lasso on the same dataset.

## Dataset

Synthetic regression data generated with `sklearn.datasets.make_regression`
(controllable noise, feature count, and effective rank).

## Key imports

```python
from rice_ml.supervised_learning import LinearRegression, RidgeRegression, LassoRegression
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import mean_squared_error, root_mean_squared_error, r2_score
```
```

---
