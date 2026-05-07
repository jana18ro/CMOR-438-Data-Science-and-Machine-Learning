# Logistic Regression

Demonstrates binary and multi-class classification using
`rice_ml.supervised_learning.LogisticRegression`.

## Notebook

`logistic_regression_notebook.ipynb`

## What the Notebook Covers

- Sigmoid function and probability interpretation of outputs.
- Cross-entropy loss and gradient descent convergence.
- Decision boundary visualisation in 2D feature space.
- Effect of `learning_rate` and `n_iterations` on training.
- Evaluation: accuracy, precision, recall, F1, ROC curve, and AUC.
- Comparison with a baseline (majority-class) classifier.

## Dataset
cancer_data.csv

## Key imports

```python
from rice_ml.supervised_learning import LogisticRegression
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, f1_score, confusion_matrix, classification_report
```
