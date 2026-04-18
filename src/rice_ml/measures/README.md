# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\measures\README.md

```markdown
# measures

Model evaluation metrics and cross-validation utilities for the `rice_ml` pipeline.
All metric functions accept NumPy arrays and return scalar floats (or DataFrames
for structured outputs like the confusion matrix).

---

## `metrics.py`

### Classification Metrics

| Function | Description |
|----------|-------------|
| `accuracy_score(y_true, y_pred)` | Fraction of correctly classified samples |
| `precision_score(y_true, y_pred, average)` | TP / (TP + FP) |
| `recall_score(y_true, y_pred, average)` | TP / (TP + FN) |
| `f1_score(y_true, y_pred, average)` | Harmonic mean of precision and recall |
| `confusion_matrix(y_true, y_pred)` | N×N count matrix |
| `classification_report(y_true, y_pred)` | Per-class precision, recall, F1, support |

The `average` parameter accepts `'binary'`, `'macro'`, `'micro'`, or `'weighted'`.

### Regression Metrics

| Function | Description |
|----------|-------------|
| `mean_squared_error(y_true, y_pred)` | Average squared residual |
| `root_mean_squared_error(y_true, y_pred)` | Square root of MSE; same units as target |
| `r2_score(y_true, y_pred)` | Coefficient of determination; 1.0 is perfect |
| `mean_absolute_error(y_true, y_pred)` | Average absolute residual |

---

## `validation.py`

### `KFoldCV(n_splits, shuffle, random_state)`
K-fold cross-validation splitter. Yields `(train_indices, val_indices)`
for each fold. Compatible with any `rice_ml` estimator.

### `cross_val_score(estimator, X, y, cv, scoring)`
Evaluates an estimator using k-fold cross-validation.
Returns a NumPy array of per-fold scores.

```python
from rice_ml.measures import cross_val_score
from rice_ml.supervised_learning import DecisionTreeClassifier

scores = cross_val_score(
    DecisionTreeClassifier(max_depth=5),
    X, y, cv=5, scoring='accuracy'
)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### `stratified_split(X, y, test_size, random_state)`
Train/test split that preserves the original class proportion in both
the training and test sets. Recommended for imbalanced classification problems.
```