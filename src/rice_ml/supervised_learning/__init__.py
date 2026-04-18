# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\supervised_learning\__init__.py

"""
rice_ml.supervised_learning
===========================
Supervised learning algorithms implemented from scratch using NumPy.

This subpackage provides estimators for both classification and regression
tasks. All classes expose a consistent scikit-learn-style interface:

    fit(X, y)       -> trains the model on labelled data
    predict(X)      -> returns predicted labels or values
    score(X, y)     -> returns a default evaluation metric (accuracy or R²)

Available Estimators
--------------------
LinearRegression
    Ordinary Least Squares regression solved via gradient descent.
    Attributes: ``weights_``, ``bias_``, ``loss_``
    Evaluation: MSE, RMSE, R²

RidgeRegression / LassoRegression
    Regularised linear regression with L2 (Ridge) and L1 (Lasso) penalties.
    Controls model complexity and prevents overfitting.

LogisticRegression
    Binary classification using sigmoid activation and cross-entropy loss,
    optimised with batch gradient descent.
    Attributes: ``weights_``, ``bias_``, ``loss_``
    Evaluation: accuracy, precision, recall, F1, ROC-AUC

KNNClassifier / KNNRegressor
    Distance-based K-Nearest Neighbours for classification and regression.
    Supports Euclidean and Manhattan distance metrics.
    Key parameter: ``n_neighbors``

DecisionTreeClassifier
    Recursive CART decision tree for classification using Gini impurity
    or information gain as the splitting criterion.
    Key parameters: ``max_depth``, ``min_samples_split``

DecisionTreeRegressor
    Recursive CART decision tree for regression using variance reduction
    as the splitting criterion.
    Key parameters: ``max_depth``, ``min_samples_split``

RandomForest / GradientBoosting
    Ensemble methods: RandomForest uses bootstrap aggregation (bagging)
    of decision trees; GradientBoosting trains trees sequentially to
    correct residual errors.

GradientDescent
    General-purpose gradient descent optimiser supporting batch, stochastic
    (SGD), and mini-batch modes. Used internally by linear models.

Perceptron
    Single-layer binary classifier using the Heaviside step activation.
    Tracks MSE loss per epoch. Includes ``plot_loss()`` and
    ``confusion_matrix()`` convenience methods.
    Key parameters: ``learning_rate``, ``n_iterations``, ``random_state``

MLP (Multi-Layer Perceptron)
    Feedforward neural network with configurable hidden layers, ReLU/sigmoid
    activations, backpropagation, dropout regularisation, and weight decay.
    Optimisers: SGD, Adam.

Examples
--------
>>> from rice_ml.supervised_learning import Perceptron
>>> import numpy as np
>>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
>>> y = np.array([0, 1, 1, 0])
>>> clf = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
>>> clf.fit(X, y)
>>> clf.predict(X)
array([0, 1, 1, 0])

>>> from rice_ml.supervised_learning import LinearRegression
>>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
>>> model.fit(X_train, y_train)
>>> print(model.score(X_test, y_test))   # returns R²
"""

from .linear_regression import LinearRegression
from .regresion import RidgeRegression, LassoRegression
from .logistic_regression import LogisticRegression
from .knn import KNNClassifier, KNNRegressor
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .ensemble import RandomForest, GradientBoosting
from .gradient_descent import GradientDescent
from .perceptron import Perceptron
from .multi_layer_perceptron import MLP

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForest",
    "GradientBoosting",
    "GradientDescent",
    "Perceptron",
    "MLP",
]
