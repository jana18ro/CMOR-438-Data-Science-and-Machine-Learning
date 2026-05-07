# Ensemble Models

# Ensembles/

Demonstrates ensemble methods — Random Forest and Gradient Boosting —
using `rice_ml.supervised_learning.RandomForest` and `GradientBoosting`,
as well as manual model stacking/blending.

## Notebook

`ensembles_notebook.ipynb`

## What the Notebook Covers

- **Bagging (Random Forest):**
  - Bootstrap sampling and subspace sampling of features.
  - How increasing `n_estimators` reduces variance.
  - Feature importance scores and visualisation.
- **Boosting (Gradient Boosting):**
  - Sequential residual correction.
  - Effect of `learning_rate` and `n_estimators` on bias-variance.
  - Comparison with a single deep decision tree.
- **Stacking / Blending:**
  - Combining predictions from multiple base learners.
  - Using a meta-learner (Logistic Regression) on out-of-fold predictions.
- **Performance Comparison:** Single tree vs. Random Forest vs.
  Gradient Boosting vs. stacked ensemble.

## Dataset

- **Classification:** Titanic survival.

## Key imports

```python
from rice_ml.supervised_learning import RandomForest, GradientBoosting, DecisionTreeClassifier
from rice_ml.processing import OrdinalEncoder, StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, r2_score, cross_val_score
```
```
