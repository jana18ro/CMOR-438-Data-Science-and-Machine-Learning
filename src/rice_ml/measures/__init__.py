"""
rice_ml.measures
================

Evaluation metrics and validation utilities for the rice_ml pipeline.

Classification metrics
----------------------
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report

Regression metrics
------------------
    mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

Validation helpers
------------------
    k_fold_indices, cross_val_score, stratified_split, holdout_evaluation

Short aliases
-------------
    accuracy, precision, recall, mse, rmse, mae
"""

from .metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    # Regression
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Short aliases
    accuracy,
    precision,
    recall,
    mse,
    rmse,
    mae,
)

from .validation import (
    k_fold_indices,
    cross_val_score,
    stratified_split,
    holdout_evaluation,
)

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
    "mean_absolute_error",
    "r2_score",
    # Aliases
    "accuracy",
    "precision",
    "recall",
    "mse",
    "rmse",
    "mae",
    # Validation
    "k_fold_indices",
    "cross_val_score",
    "stratified_split",
    "holdout_evaluation",
]
