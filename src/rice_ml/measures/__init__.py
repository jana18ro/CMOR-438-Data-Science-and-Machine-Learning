# FILE: 2026_Data_Science_and_Machine_Learning\src\rice_ml\measures\__init__.py

"""
rice_ml.measures
================
Model evaluation metrics and cross-validation utilities for the rice_ml pipeline.

This subpackage provides functions to assess model quality on both classification
and regression tasks, as well as tools for robust model selection via cross-validation.

Metrics (metrics.py)
---------------------
Classification metrics:
    accuracy_score(y_true, y_pred)
        Fraction of correctly classified samples.

    precision_score(y_true, y_pred, average)
        Ratio of true positives to all predicted positives.

    recall_score(y_true, y_pred, average)
        Ratio of true positives to all actual positives.

    f1_score(y_true, y_pred, average)
        Harmonic mean of precision and recall.

    confusion_matrix(y_true, y_pred)
        Returns an N×N array of prediction vs. actual counts.

    classification_report(y_true, y_pred)
        Prints precision, recall, F1, and support for each class.

Regression metrics:
    mean_squared_error(y_true, y_pred)   -> MSE
    root_mean_squared_error(y_true, y_pred) -> RMSE
    r2_score(y_true, y_pred)             -> coefficient of determination R²
    mean_absolute_error(y_true, y_pred)  -> MAE

Validation (validation.py)
---------------------------
KFoldCV(n_splits, shuffle, random_state)
    K-fold cross-validation splitter. Yields (train_indices, val_indices)
    for each fold. Use with any rice_ml estimator.

cross_val_score(estimator, X, y, cv, scoring)
    Evaluates an estimator using cross-validation and returns per-fold scores.

stratified_split(X, y, test_size, random_state)
    Stratified train/test split preserving class proportions.

Examples
--------
>>> from rice_ml.measures import accuracy_score, confusion_matrix, r2_score
>>> print(accuracy_score(y_test, y_pred))
>>> print(confusion_matrix(y_test, y_pred))

>>> from rice_ml.measures import cross_val_score
>>> from rice_ml.supervised_learning import LogisticRegression
>>> scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='accuracy')
>>> print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
"""

from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from .validation import KFoldCV, cross_val_score, stratified_split

__all__ = [
    # Classification
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "classification_report",
    # Regression
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "mean_absolute_error",
    # Validation
    "KFoldCV",
    "cross_val_score",
    "stratified_split",
]
