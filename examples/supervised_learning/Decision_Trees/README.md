# Decision Trees (Classifier and Regressor)

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

# Decision Tree Regressor — From Scratch

A **pure-NumPy implementation** of a Decision Tree Regressor, built without scikit-learn. This notebook walks through the theory, training, evaluation, and comparison against scikit-learn's own implementation.

---

## Overview

This project demonstrates how a decision tree regressor works under the hood by building one from scratch using only NumPy. It covers everything from the core splitting criterion to bias–variance tradeoff analysis and feature importance.

---

## Notebook Structure

| Section | Description |
|---|---|
| **1. Source Code** | Imports the custom `decision_tree_regressor` class from the `rice_ml` package |
| **2. Imports & Setup** | Loads NumPy, Matplotlib, Seaborn, and scikit-learn utilities |
| **3. Intuition** | Explains variance reduction and recursive splitting with visualisations |
| **4. Dataset** | Generates and explores a synthetic 8-feature regression dataset |
| **5. Training & Evaluation** | Fits the custom model, reports R² and MSE, plots residuals |
| **6. Effect of `max_depth`** | Sweeps depth values to illustrate the bias–variance tradeoff |
| **7. Step-Function Behaviour** | 1-D visualisation of how tree depth affects prediction smoothness |
| **8. Feature Importance** | Permutation-based feature importance |
| **9. sklearn Comparison** | Side-by-side benchmark against `sklearn.tree.DecisionTreeRegressor` |
| **10. Non-linear Demo** | Fits a noisy `sin(x)` curve to demonstrate step-function approximation |
| **11. Summary** | Key results and takeaways |

---

## How It Works

A regression tree recursively partitions the input space into rectangular regions. Within each region (leaf), the prediction is the **mean** of the training targets that fall in that leaf.

### Splitting Criterion — Variance Reduction

At each node, the split that maximally reduces variance is chosen:

$$\text{Variance} = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2$$

$$\text{Variance Reduction} = \text{Var}(\text{parent}) - \frac{|L|}{|P|}\text{Var}(L) - \frac{|R|}{|P|}\text{Var}(R)$$

The result is a **step-function approximation** of the true relationship, where `max_depth` controls the trade-off between bias and variance.

---

## Dataset

A synthetic regression dataset is generated using `sklearn.datasets.make_regression`:

- **600 samples**, **8 features**, noise level = 20
- **Train / Test split**: 80% / 20%
- Random seed: 42 (for reproducibility)

---

## Key Results

| Model | Train R² | Test R² | Implementation |
|---|---|---|---|
| **Our DTR** (depth=6) | ≥ 0.95 | ≥ 0.70 | Pure NumPy |
| **sklearn DTR** (depth=6) | ≥ 0.95 | ≥ 0.70 | Cython / CART |

Our from-scratch implementation closely matches scikit-learn on standard regression benchmarks.

---

## Usage

1. Ensure the `rice_ml` package path is correctly set in Section 1 of the notebook.
2. Run all cells in order from top to bottom.
3. All plots are generated inline.

---

## Key Takeaways

- Decision tree regressors split data to maximally reduce **variance** within leaves.
- Each leaf predicts the **mean** of the training samples that fall in it.
- The output is a **step-function approximation** of the true relationship.
- `max_depth` controls the **bias–variance tradeoff**: deeper trees have lower bias but higher variance.
- The from-scratch NumPy implementation achieves performance comparable to scikit-learn.

```
