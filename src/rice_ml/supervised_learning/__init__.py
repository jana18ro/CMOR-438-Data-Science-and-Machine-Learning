"""
rice_ml.supervised_learning
===========================
Supervised learning algorithms implemented from scratch using NumPy.

This subpackage provides estimators for both classification and regression
tasks. All classes expose a consistent interface:

    fit(X, y)       -> trains the model on labelled data
    predict(X)      -> returns predicted labels or values
    score(X, y)     -> returns a default evaluation metric (accuracy or R²)

Available Estimators
--------------------
SimpleLinearRegression
    Single-feature Linear Regression via gradient descent.
    Attributes: ``coef_``, ``intercept_``, ``cost_history_``

LogisticRegression
    Binary classification using sigmoid activation and cross-entropy loss,
    optimised with batch gradient descent.

KNN
    K-Nearest Neighbours lazy classifier.

decision_tree_classifier
    Recursive CART decision tree for classification using information gain.

decision_tree_regressor
    Recursive CART decision tree for regression using variance reduction.

hard_voting_classifier / bagging_classifier / random_forest_classifier
    Ensemble methods from ensemble.py.

GradientDescent
    General-purpose gradient descent optimiser.

Perceptron
    Single-layer binary classifier with Heaviside activation.

multi_layer_perceptron
    Feedforward neural network with configurable hidden layers.
"""

from .linear_regression import SimpleLinearRegression
from .logistic_regression import LogisticRegression
from .knn import KNN
from .decision_tree_classifier import decision_tree_classifier
from .decision_tree_regressor import decision_tree_regressor
from .ensemble import (
    hard_voting_classifier,
    bagging_classifier,
    random_forest_classifier,
    _ensemble_tree_classifier,
)
from .gradient_descent import GradientDescent
from .perceptron import Perceptron
from .multi_layer_perceptron import multi_layer_perceptron

# Convenience aliases used in notebooks / docs
LinearRegression = SimpleLinearRegression
KNNClassifier = KNN
KNNRegressor = KNN
DecisionTreeClassifier = decision_tree_classifier
DecisionTreeRegressor = decision_tree_regressor
RandomForest = random_forest_classifier
GradientBoosting = bagging_classifier
MLP = multi_layer_perceptron

__all__ = [
    "SimpleLinearRegression",
    "LinearRegression",
    "LogisticRegression",
    "KNN",
    "KNNClassifier",
    "KNNRegressor",
    "decision_tree_classifier",
    "decision_tree_regressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "hard_voting_classifier",
    "bagging_classifier",
    "random_forest_classifier",
    "RandomForest",
    "GradientBoosting",
    "_ensemble_tree_classifier",
    "GradientDescent",
    "Perceptron",
    "multi_layer_perceptron",
    "MLP",
]
