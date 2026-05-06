"""
logistic_regression.py

A minimal, from-scratch implementation of Logistic Regression using
batch gradient descent. Supports probability prediction, class prediction,
and accuracy scoring. This implementation is intended for educational
purposes and mirrors the behavior of sklearn's LogisticRegression
(without regularization or solvers).

The model optimizes the binary cross-entropy loss:

    L = −[ y log(p) + (1 − y) log(1 − p) ]

Typical usage
-------------
model = LogisticRegression(learning_rate=0.01, n_iterations=5000)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
preds = model.predict(X_test)
acc = model.score(X_test, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    A custom Logistic Regression classifier.
    
    This class implements binary classification using the sigmoid 
    activation function and binary cross-entropy loss, optimized 
    via batch gradient descent.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize model parameters.
        
        Args:
            learning_rate (float): The step size for weight updates.
            n_iterations (int): Total number of gradient descent passes.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Compute the sigmoid of z.
        
        Uses clipping to prevent overflow in the exponential function.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        """
        Calculate the mean binary cross-entropy loss.
        """
        # Small epsilon added to avoid log(0) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Fit the model using batch gradient descent.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target binary labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # Calculate the pre-activation (linear combination)
            linear_output = np.dot(X, self.weights) + self.bias
            
            # Apply activation function
            y_hat = self._sigmoid(linear_output)

            # Compute gradients for weights and bias
            # Gradient = (1/n) * X_transpose * (predictions - actual)
            error = y_hat - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Log loss periodically for monitoring
            if (i + 1) % (self.n_iterations // 10 or 1) == 0:
                current_loss = self._compute_loss(y, y_hat)
                print(f"Iteration {i + 1}/{self.n_iterations} - Loss: {current_loss:.6f}")

    def predict_proba(self, X):
        """
        Predict probability of the positive class (1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X):
        """
        Predict binary labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        """
        Return the mean accuracy of the predictions.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)