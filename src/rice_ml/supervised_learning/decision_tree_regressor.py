"""
decision_tree_classifier.py

An educational implementation of a Decision Tree Classifier using NumPy.

This file provides a from-scratch classifier for supervised learning problems
where the target variable contains categories or class labels. The model learns
a set of decision rules by repeatedly splitting the training data into smaller
groups. At each split, the algorithm chooses the feature and threshold that
produce the greatest improvement in class purity.

The classifier uses:
- binary splits of the form: feature <= threshold
- entropy to measure class impurity
- information gain to choose the best split
- recursive tree construction
- majority-class prediction at leaf nodes

This implementation is intended for learning how decision trees work internally.
It does not rely on scikit-learn.

Example
-------
>>> import numpy as np
>>> from decision_tree_classifier import decision_tree_classifier
>>>
>>> X = np.array([[0, 0],
...               [0, 1],
...               [1, 0],
...               [1, 1]])
>>>
>>> y = np.array([0, 0, 1, 1])
>>>
>>> tree = decision_tree_classifier(max_depth=2)
>>> tree.fit(X, y)
>>> tree.predict(X)
array([0, 0, 1, 1])
"""

import numpy as np


class decision_tree_classifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Create a decision tree classifier.

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
        self.classes_ = None

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
        Convert y into a one-dimensional NumPy array.
        """
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one value.")

        return y

    # ---------------------------------------------------------
    # Entropy and information gain
    # ---------------------------------------------------------

    def _entropy(self, y):
        """
        Calculate entropy.

        Entropy measures how mixed the class labels are.

        A node with only one class has entropy 0.
        A node with many evenly mixed classes has higher entropy.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Calculate how much a split improves class purity.

        Information gain compares the entropy before the split to the weighted
        entropy after the split.
        """
        parent_entropy = self._entropy(parent_y)

        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        children_entropy = (
            left_weight * self._entropy(left_y)
            + right_weight * self._entropy(right_y)
        )

        return parent_entropy - children_entropy

    # ---------------------------------------------------------
    # Finding the best split
    # ---------------------------------------------------------

    def _best_split(self, X, y):
        """
        Search all features and thresholds to find the best split.

        Returns
        -------
        best_feature : int or None
            The index of the best feature.

        best_threshold : float or None
            The threshold value for the best split.

        best_gain : float
            The information gain from the best split.
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_gain = 0

        for feature_index in range(n_features):
            possible_thresholds = np.unique(X[:, feature_index])

            for threshold in possible_thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    # ---------------------------------------------------------
    # Leaf prediction
    # ---------------------------------------------------------

    def _majority_class(self, y):
        """
        Return the most common class label in a node.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ---------------------------------------------------------
    # Building the tree
    # ---------------------------------------------------------

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

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
            "prediction": class_label
        }
        """
        n_samples = X.shape[0]
        unique_classes = np.unique(y)

        pure_node = len(unique_classes) == 1
        too_small = n_samples < self.min_samples_split
        reached_max_depth = (
            self.max_depth is not None and depth >= self.max_depth
        )

        if pure_node or too_small or reached_max_depth:
            return {"prediction": self._majority_class(y)}

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            return {"prediction": self._majority_class(y)}

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
        Train the decision tree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
            The fitted classifier.
        """
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)

        return self

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def _predict_one(self, sample, node):
        """
        Predict the class label for one sample.
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
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted class labels.
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
        Return classification accuracy.

        Accuracy = number of correct predictions / total predictions.
        """
        y = self._prepare_y(y)
        predictions = self.predict(X)

        if len(predictions) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        return np.mean(predictions == y)

