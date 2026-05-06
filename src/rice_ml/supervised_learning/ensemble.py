"""
ensemble.py

Educational implementations of common ensemble classification methods.

This file builds several ensemble learners from scratch using NumPy. Ensemble
methods combine the predictions of multiple smaller models in order to produce
a stronger final prediction than one model may produce alone.

This module includes:

- hard_voting_classifier:
    Combines several fitted classifiers and predicts the class that receives
    the most votes.

- bagging_classifier:
    Trains many decision trees on different bootstrap samples of the training
    data, then combines their predictions by majority vote.

- random_forest_classifier:
    Extends bagging by training many decision trees on bootstrap samples while
    also limiting the number of features considered at each split.

The decision tree used inside the bagging and random forest models is included
inside this file, so the ensemble methods can run independently without relying
on scikit-learn.

The goal of this file is to show how ensemble methods work internally in a
clear and readable way.
"""

import numpy as np


# ============================================================
# Helper functions
# ============================================================

def _prepare_X(X):
    """
    Convert input features into a two-dimensional NumPy array.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    return X


def _prepare_y(y):
    """
    Convert target labels into a one-dimensional NumPy array.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if y.shape[0] == 0:
        raise ValueError("y must contain at least one value.")

    return y


# ============================================================
# Internal decision tree classifier
# ============================================================

class _ensemble_tree_classifier:
    """
    A small decision tree classifier used internally by ensemble models.

    The tree uses:
    - entropy to measure impurity
    - information gain to select splits
    - binary splits of the form feature <= threshold
    - majority class labels at leaf nodes

    This class is intentionally private because it exists mainly as the base
    learner for bagging and random forest models in this file.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.tree = None
        self.rng = np.random.default_rng(random_state)

    def _entropy(self, y):
        """
        Calculate entropy for a group of class labels.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Calculate how much a split reduces class impurity.
        """
        parent_entropy = self._entropy(parent_y)

        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        child_entropy = (
            left_weight * self._entropy(left_y)
            + right_weight * self._entropy(right_y)
        )

        return parent_entropy - child_entropy

    def _majority_class(self, y):
        """
        Return the most common class label in a node.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _choose_feature_indices(self, n_features):
        """
        Decide which features are available for splitting at a node.
        """
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            number_to_select = min(self.max_features, n_features)

        elif self.max_features == "sqrt":
            number_to_select = max(1, int(np.sqrt(n_features)))

        elif self.max_features == "log2":
            number_to_select = max(1, int(np.log2(n_features)))

        elif isinstance(self.max_features, float):
            number_to_select = max(1, int(self.max_features * n_features))

        else:
            raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'.")

        return self.rng.choice(
            n_features,
            size=number_to_select,
            replace=False
        )

    def _best_split(self, X, y):
        """
        Search for the feature and threshold with the highest information gain.
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_gain = 0

        candidate_features = self._choose_feature_indices(n_features)

        for feature_index in candidate_features:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
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

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
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

    def fit(self, X, y):
        """
        Train the internal decision tree.
        """
        X = _prepare_X(X)
        y = _prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.tree = self._build_tree(X, y, depth=0)

        return self

    def _predict_one(self, sample, node):
        """
        Predict one sample by following the tree branches.
        """
        if "prediction" in node:
            return node["prediction"]

        feature = node["feature"]
        threshold = node["threshold"]

        if sample[feature] <= threshold:
            return self._predict_one(sample, node["left"])

        return self._predict_one(sample, node["right"])

    def predict(self, X):
        """
        Predict class labels.
        """
        if self.tree is None:
            raise RuntimeError("The tree must be fitted before prediction.")

        X = _prepare_X(X)

        return np.array([
            self._predict_one(sample, self.tree)
            for sample in X
        ])


# ============================================================
# Hard Voting Classifier
# ============================================================

class hard_voting_classifier:
    """
    A hard voting classifier.

    This ensemble receives a list of already-created classifiers. Each classifier
    must have a fit method and a predict method.

    During prediction, every classifier votes for a class label. The final
    prediction is the class label with the most votes.
    """

    def __init__(self, classifiers):
        """
        Parameters
        ----------
        classifiers : list
            A list of classifier objects.

            Each classifier should have:
            - fit(X, y)
            - predict(X)
        """
        if len(classifiers) == 0:
            raise ValueError("At least one classifier must be provided.")

        self.classifiers = classifiers
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit each classifier on the same training data.
        """
        X = _prepare_X(X)
        y = _prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)

        for classifier in self.classifiers:
            classifier.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict class labels by majority vote.
        """
        if self.classes_ is None:
            raise RuntimeError("The voting classifier must be fitted first.")

        X = _prepare_X(X)

        all_predictions = np.array([
            classifier.predict(X)
            for classifier in self.classifiers
        ])

        final_predictions = []

        for sample_predictions in all_predictions.T:
            labels, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(labels[np.argmax(counts)])

        return np.array(final_predictions)

    def score(self, X, y):
        """
        Return classification accuracy.
        """
        y = _prepare_y(y)
        predictions = self.predict(X)

        return np.mean(predictions == y)


# ============================================================
# Bagging Classifier
# ============================================================

class bagging_classifier:
    """
    A bagging classifier using decision trees as base learners.

    Bagging means bootstrap aggregating.

    The process is:
    1. Draw many bootstrap samples from the training data.
    2. Train one decision tree on each bootstrap sample.
    3. Let all trees vote during prediction.
    4. Use the majority vote as the final prediction.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        bootstrap=True,
        random_state=None
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the ensemble.

        max_depth : int or None
            Maximum depth of each tree.

        min_samples_split : int
            Minimum number of samples needed to split a node.

        bootstrap : bool
            If True, sample rows with replacement.

        random_state : int or None
            Controls reproducibility.
        """
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Train the bagging ensemble.
        """
        X = _prepare_X(X)
        y = _prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.trees = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                row_indices = rng.integers(
                    low=0,
                    high=n_samples,
                    size=n_samples
                )
            else:
                row_indices = rng.permutation(n_samples)

            X_sample = X[row_indices]
            y_sample = y[row_indices]

            tree = _ensemble_tree_classifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=None,
                random_state=rng.integers(0, 2**31)
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict by majority vote across all trees.
        """
        if len(self.trees) == 0:
            raise RuntimeError("The bagging classifier must be fitted first.")

        X = _prepare_X(X)

        all_predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        final_predictions = []

        for sample_predictions in all_predictions.T:
            labels, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(labels[np.argmax(counts)])

        return np.array(final_predictions)

    def score(self, X, y):
        """
        Return classification accuracy.
        """
        y = _prepare_y(y)
        predictions = self.predict(X)

        return np.mean(predictions == y)


# ============================================================
# Random Forest Classifier
# ============================================================

class random_forest_classifier:
    """
    A random forest classifier.

    A random forest is similar to bagging, but each decision tree is also given
    only a random subset of features to consider when looking for the best split.

    This added randomness helps make the trees less similar to each other.
    When the trees are less similar, their combined vote is often stronger.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=None
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.

        max_depth : int or None
            Maximum depth of each tree.

        min_samples_split : int
            Minimum number of samples required to split a node.

        max_features : None, int, float, "sqrt", or "log2"
            Number of features considered at each split.

        bootstrap : bool
            If True, each tree trains on a bootstrap sample.

        random_state : int or None
            Controls reproducibility.
        """
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Train the random forest.
        """
        X = _prepare_X(X)
        y = _prepare_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.trees = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                row_indices = rng.integers(
                    low=0,
                    high=n_samples,
                    size=n_samples
                )
            else:
                row_indices = rng.permutation(n_samples)

            X_sample = X[row_indices]
            y_sample = y[row_indices]

            tree = _ensemble_tree_classifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.integers(0, 2**31)
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict by majority vote across all trees.
        """
        if len(self.trees) == 0:
            raise RuntimeError("The random forest classifier must be fitted first.")

        X = _prepare_X(X)

        all_predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        final_predictions = []

        for sample_predictions in all_predictions.T:
            labels, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(labels[np.argmax(counts)])

        return np.array(final_predictions)

    def score(self, X, y):
        """
        Return classification accuracy.
        """
        y = _prepare_y(y)
        predictions = self.predict(X)

        return np.mean(predictions == y)