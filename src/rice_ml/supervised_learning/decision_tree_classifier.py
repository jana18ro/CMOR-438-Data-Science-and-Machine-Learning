"""
decision_tree_regressor.py

An educational implementation of a Decision Tree Regressor using NumPy.

This file provides a from-scratch regressor for supervised learning problems
where the target variable is numerical. The model learns decision rules by
recursively splitting the training data into smaller groups. At each split, the
algorithm chooses the feature and threshold that reduce target-value variance
the most.

The regressor uses:
- binary splits of the form: feature <= threshold
- variance as the measure of target spread
- variance reduction to choose the best split
- recursive tree construction
- mean target value as the prediction at leaf nodes

This implementation is intended for learning how regression trees work
internally. It does not rely on scikit-learn.

Example
-------
>>> import numpy as np
>>> from decision_tree_regressor import decision_tree_regressor
>>>
>>> X = np.array([[0],
...               [1],
...               [2],
...               [3]])
>>>
>>> y = np.array([0.0, 1.0, 4.0, 9.0])
>>>
>>> tree = decision_tree_regressor(max_depth=2)
>>> tree.fit(X, y)
>>> tree.predict(X)
array([0. , 1. , 4. , 9. ])
"""

import numpy as np


class decision_tree_regressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Create a decision tree regressor.

        Parameters
        ----------
        max_depth : int or None
            The maximum depth allowed for the tree.

            If max_depth=None, the tree can keep growing until stopping
            conditions are met.

        min_samples_split : int
            The minimum number of samples needed at a node before the model
            is allowed to split that node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # ---------------------------------------------------------
    # Data checking
    # ---------------------------------------------------------

    def _prepare_X(self, X):
        """
        Convert X into a two-dimensional NumPy array.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        return X

    def _prepare_y(self, y):
        """
        Convert y into a one-dimensional numeric NumPy array.
        """
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one value.")

        return y

    # ---------------------------------------------------------
    # Variance and variance reduction
    # ---------------------------------------------------------

    def _variance(self, y):
        """
        Calculate variance of target values.

        Variance tells us how spread out the numerical target values are.
        A good regression-tree split should reduce this spread.
        """
        if len(y) == 0:
            return 0.0

        return np.mean((y - np.mean(y)) ** 2)

    def _variance_reduction(self, parent_y, left_y, right_y):
        """
        Calculate how much a split reduces variance.

        The model compares the variance before the split to the weighted
        variance after the split.
        """
        parent_variance = self._variance(parent_y)

        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        children_variance = (
            left_weight * self._variance(left_y)
            + right_weight * self._variance(right_y)
        )

        return parent_variance - children_variance

    # ---------------------------------------------------------
    # Finding the best split
    # ---------------------------------------------------------

    def _best_split(self, X, y):
        """
        Search all features and thresholds to find the split with the greatest
        variance reduction.

        Returns
        -------
        best_feature : int or None
            The index of the best feature.

        best_threshold : float or None
            The threshold value for the best split.

        best_reduction : float
            The variance reduction from the best split.
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_reduction = 0

        for feature_index in range(n_features):
            possible_thresholds = np.unique(X[:, feature_index])

            for threshold in possible_thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                reduction = self._variance_reduction(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_reduction

    # ---------------------------------------------------------
    # Leaf prediction
    # ---------------------------------------------------------

    def _leaf_value(self, y):
        """
        Return the prediction for a leaf node.

        For regression trees, the leaf prediction is the average target value
        of the training examples that reached that leaf.
        """
        return float(np.mean(y))

    # ---------------------------------------------------------
    # Building the tree
    # ---------------------------------------------------------

    def _build_tree(self, X, y, depth):
        """
        Recursively build the regression tree.

        A tree node is stored as a dictionary.

        Internal node format:
        {
            "feature": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

        Leaf node format:
        {
            "prediction": numerical_value
        }
        """
        n_samples = X.shape[0]

        same_target_value = len(np.unique(y)) == 1
        too_small = n_samples < self.min_samples_split
        reached_max_depth = (
            self.max_depth is not None and depth >= self.max_depth
        )

        if same_target_value or too_small or reached_max_depth:
            return {"prediction": self._leaf_value(y)}

        feature, threshold, reduction = self._best_split(X, y)

        if feature is None or reduction <= 0:
            return {"prediction": self._leaf_value(y)}

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_branch = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1
        )

        right_branch = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1
        )

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_branch,
            "right": right_branch
        }

    # ---------------------------------------------------------
    # Public training method
    # ---------------------------------------------------------

    def fit(self, X, y):
        """
        Train the decision tree regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Numerical target values.

        Returns
        -------
        self
            The fitted regressor.
        """
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.tree = self._build_tree(X, y, depth=0)

        return self

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def _predict_one(self, sample, node):
        """
        Predict the numerical target value for one sample.
        """
        if "prediction" in node:
            return node["prediction"]

        feature = node["feature"]
        threshold = node["threshold"]

        if sample[feature] <= threshold:
            return self._predict_one(sample, node["left"])
        else:
            return self._predict_one(sample, node["right"])

    def predict(self, X):
        """
        Predict numerical target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if self.tree is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X = self._prepare_X(X)

        return np.array([
            self._predict_one(sample, self.tree)
            for sample in X
        ])

    def score(self, X, y):
        """
        Return the R-squared score.

        R-squared measures how much of the variation in y is explained by the
        model. A score closer to 1 is better.
        """
        y = self._prepare_y(y)
        predictions = self.predict(X)

        if len(predictions) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        ss_residual = np.sum((y - predictions) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)

        if ss_total == 0:
            return 1.0 if ss_residual == 0 else 0.0

        return 1 - (ss_residual / ss_total)

    def mean_squared_error(self, X, y):
        """
        Return mean squared error.

        MSE is the average squared difference between true values and predicted
        values. A smaller MSE is better.
        """
        y = self._prepare_y(y)
        predictions = self.predict(X)

        return np.mean((y - predictions) ** 2)