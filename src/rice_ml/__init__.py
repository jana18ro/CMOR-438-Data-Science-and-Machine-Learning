# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\__init__.py

"""
rice_ml
=======
A from-scratch machine learning library developed for CMOR 438
(Data Science and Machine Learning) at Rice University, 2026.

This package provides clean, NumPy-based implementations of supervised and
unsupervised learning algorithms, along with data processing utilities and
evaluation metrics — all built without relying on scikit-learn internally.

Subpackages
-----------
supervised_learning
    Classifiers and regressors: Linear Regression, Logistic Regression,
    KNN, Decision Trees, Regression Trees, Ensemble Methods, Perceptron, MLP.

unsupervised_learning
    Clustering and dimensionality reduction: K-Means, DBSCAN, PCA, SVD,
    Label Propagation / Community Detection.

processing
    Data preparation utilities: StandardScaler, MinMaxScaler, OrdinalEncoder,
    train_test_split, postprocessing helpers.

measures
    Evaluation metrics and validation: accuracy, precision, recall, F1,
    MSE, RMSE, R², confusion matrix, k-fold cross-validation.

Design Principles
-----------------
- Consistent ``fit`` / ``predict`` / ``score`` interface across all estimators.
- Reproducibility via ``random_state`` parameters throughout.
- Pure NumPy inside algorithm code — scikit-learn used only for datasets
  and external metric comparisons in notebooks.
- Vectorized operations for performance wherever possible.

Examples
--------
>>> from rice_ml.supervised_learning import Perceptron
>>> from rice_ml.processing import StandardScaler, train_test_split
>>> from rice_ml.measures import accuracy_score
>>> clf = Perceptron(learning_rate=0.1, n_iterations=200, random_state=42)
>>> clf.fit(X_train, y_train)
>>> print(accuracy_score(y_test, clf.predict(X_test)))
"""

from rice_ml import supervised_learning
from rice_ml import unsupervised_learning
from rice_ml import processing
from rice_ml import measures

__version__ = "0.1.0"
__author__  = "Jana"
__license__ = "MIT"

__all__ = [
    "supervised_learning",
    "unsupervised_learning",
    "processing",
    "measures",
]
