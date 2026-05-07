# Decision Trees Classifier

# Decision_Trees/

Demonstrates the CART decision tree classifier using
`rice_ml.supervised_learning.DecisionTreeClassifier`.

## Notebook

`decision_tree_classifier_notebook.ipynb`

## What the Notebook Covers

- Gini impurity and information gain as splitting criteria — comparison of both.
- Recursive partitioning: how the tree depth affects decision boundary complexity.
- Visualisation of the learned tree structure (nodes, branches, leaf labels).
- Effect of `max_depth` on train vs. test accuracy (overfitting demonstration).
- Feature importance derived from split counts / impurity reduction.
- Evaluation: accuracy, precision, recall, F1, confusion matrix.

## Dataset

- **Titanic** survival dataset (binary classification).
- **Iris** (multi-class classification).

## Key imports

```python
from rice_ml.supervised_learning import DecisionTreeClassifier
from rice_ml.processing import OrdinalEncoder, train_test_split
from rice_ml.measures import accuracy_score, confusion_matrix, classification_report
```
```
