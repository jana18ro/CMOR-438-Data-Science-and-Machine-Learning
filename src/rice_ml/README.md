
# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\README.md

```markdown
# rice_ml

`rice_ml` is a from-scratch machine learning library implementing a comprehensive
set of supervised and unsupervised learning algorithms using Python and NumPy.
It was developed as the core deliverable for **CMOR 438 (Data Science and Machine
Learning) at Rice University**.

---

## Package Overview

```
rice_ml/
├── supervised_learning/      # Classifiers and regressors
│   ├── linear_regression.py
│   ├── regresion.py          # Ridge and Lasso
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree_classifier.py
│   ├── decision_tree_regressor.py
│   ├── ensemble.py
│   ├── gradient_descent.py
│   ├── perceptron.py
│   ├── multi_layer_perceptron.py
│   └── __init__.py
├── unsupervised_learning/    # Clustering and dimensionality reduction
│   ├── kmeans.py
│   ├── dbscan.py
│   ├── pca.py
│   ├── svd.py
│   ├── label_propagation_community_detection.py
│   └── __init__.py
├── processing/               # Data preparation utilities
│   ├── preprocess.py
│   ├── postprocess.py
│   └── __init__.py
├── measures/                 # Metrics and validation
│   ├── metrics.py
│   ├── validation.py
│   └── __init__.py
└── __init__.py
```

---

## Submodules

### [`supervised_learning/`](supervised_learning/README.md)
Implements predictive models for labelled data. Covers linear models, tree-based methods,
k-nearest neighbours, neural networks (Perceptron and MLP), and ensemble techniques.

### [`unsupervised_learning/`](unsupervised_learning/README.md)
Implements models for pattern discovery in unlabelled data. Covers K-Means, DBSCAN, PCA,
SVD, and Label Propagation for community detection.

### [`processing/`](processing/README.md)
Provides data preparation utilities: feature scalers (`StandardScaler`, `MinMaxScaler`),
categorical encoders (`OrdinalEncoder`), a `train_test_split` function, and
postprocessing helpers for formatting predictions and decoding labels.

### [`measures/`](measures/README.md)
Provides evaluation metrics for both classification (accuracy, precision, recall, F1,
confusion matrix) and regression (MSE, RMSE, R², MAE), plus cross-validation utilities.

---

## Consistent API

All estimators in `rice_ml` follow the same interface pattern:

```python
model = SomeEstimator(param1=value1, param2=value2, random_state=42)
model.fit(X_train, y_train)        # learns parameters from training data
predictions = model.predict(X_test) # returns labels or values
score = model.score(X_test, y_test) # returns default metric (accuracy or R²)
```

---

## Quick Import Reference

```python
# Supervised learning
from rice_ml.supervised_learning import (
    LinearRegression, RidgeRegression, LassoRegression,
    LogisticRegression, KNNClassifier, KNNRegressor,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForest, GradientBoosting,
    Perceptron, MLP
)

# Unsupervised learning
from rice_ml.unsupervised_learning import KMeans, DBSCAN, PCA, SVD, LabelPropagation

# Processing
from rice_ml.processing import StandardScaler, MinMaxScaler, OrdinalEncoder, train_test_split

# Metrics
from rice_ml.measures import accuracy_score, f1_score, r2_score, cross_val_score
```
```

---