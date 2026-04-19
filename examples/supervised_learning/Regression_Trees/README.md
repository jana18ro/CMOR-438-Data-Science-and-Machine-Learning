# Regression Trees Notebook

```markdown
# Regression_Trees/

Demonstrates the CART decision tree regressor using
`rice_ml.supervised_learning.DecisionTreeRegressor`.

## Notebook

`regression_tree_notebook.ipynb`

## What the Notebook Covers

- Variance reduction as the splitting criterion for regression tasks.
- How `max_depth` controls the smoothness of the predicted function —
  shallow trees underfit, deep trees overfit.
- Visualisation: predicted step-function vs. true underlying curve.
- Comparison with `LinearRegression` on linear vs. non-linear datasets.
- Evaluation: MSE, RMSE, R².

## Dataset

Synthetic non-linear regression data (sinusoidal with Gaussian noise).

## Key imports

```python
from rice_ml.supervised_learning import DecisionTreeRegressor, LinearRegression
from rice_ml.processing import train_test_split
from rice_ml.measures import mean_squared_error, root_mean_squared_error, r2_score
```
```

---
