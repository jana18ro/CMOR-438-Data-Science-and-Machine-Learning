"""
perceptron.py
A from-scratch implementation of a Single Perceptron binary classifier.

Algorithm Summary
1. Initialise weights to small random values and bias to zero.
2. For each training sample (x_i, y_i):
   a. Compute the linear combination: z = dot(weights, x_i) + bias
   b. Apply the step (Heaviside) activation: y_hat = 1 if z >= 0 else 0
   c. Compute the error: delta = learning_rate * (y_i - y_hat)
   d. Update weights: weights += delta * x_i
   e. Update bias:    bias    += delta
3. Repeat for n_iterations epochs.
4. Track mean-squared-error (MSE) loss after every epoch.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Perceptron:
    """
    Single Perceptron binary classifier.

    Parameters
    learning_rate : float, default=0.01
        Step size (η) applied to each weight update. 

    n_iterations : int, default=1000
        Number of full passes (epochs) over the training data.  T

    random_state : int or None, default=42
        Seed for NumPy's random number generator, ensuring reproducible weight
        initialisation.  

    Attributes
    weights_ : np.ndarray of shape (n_features,) (Learned weight vector after fitting.  None before ``fit`` is called.)
    bias_ : float (Learned scalar bias term after fitting.  Initialised to 0.)
    loss_ : list of float (Mean squared error (MSE) recorded at the end of every epoch.)
    n_features_in_ : int (Number of features seen during ``fit``.)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Initialised by fit()
        self.weights_ = None
        self.bias_ = None
        self.loss_ = []
        self.n_features_in_ = None

    def _step_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the sign activation function.

        Parameters - z : np.ndarray 
                        Linear combination of inputs and weights (pre-activation values).

        Returns - np.ndarray of int
                        Binary predictions (0 or 1) for each element of ``z``.
        """
        return np.where(z >= 0, 1, 0)

    def _linear_output(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination z = X @ weights_ + bias_.

        Parameters X - np.ndarray of shape (n_samples, n_features)
                    Input feature matrix.

        Returns - np.ndarray of shape (n_samples,)
                    Pre-activation (raw) scores for each sample.
        """
        return np.dot(X, self.weights_) + self.bias_

    def _mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean squared error (MSE) between true and predicted labels.

        Parameters 
        y_true : np.ndarray of shape (n_samples,)
                Ground-truth binary labels.
        y_pred : np.ndarray of shape (n_samples,)
                Predicted binary labels.

        Returns - float
                    Scalar MSE loss value for the current epoch.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the Perceptron on labelled data.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.  Must be numeric; apply preprocessing
            (e.g. scaling, TF-IDF) before calling fit.
        y : np.ndarray of shape (n_samples,)
            Binary target labels.  Values should be 0 or 1.

        Returns
        self : Perceptron
            Fitted estimator (enables method chaining, e.g. ``clf.fit(X, y).predict(X)``).

        Notes:
        - Weights are initialised from a uniform distribution on [0, 1) using ``random_state`` for reproducibility.
        - MSE loss is recorded after each full epoch (not per sample).
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} labels."
            )

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Initialise parameters
        self.weights_ = rng.uniform(0, 1, size=n_features)
        self.bias_ = 0.0
        self.loss_ = []

        for _ in range(self.n_iterations):
            for idx in range(n_samples):
                x_i = X[idx]
                z = np.dot(x_i, self.weights_) + self.bias_
                y_hat = 1 if z >= 0 else 0

                delta = self.learning_rate * (y[idx] - y_hat)
                self.weights_ += delta * x_i
                self.bias_ += delta

            epoch_preds = self._step_activation(self._linear_output(X))
            self.loss_.append(self._mse_loss(y, epoch_preds))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples.

        Parameters - X : np.ndarray of shape (n_samples, n_features)

        Returns - np.ndarray of shape (n_samples,) and dtype int
                    Predicted binary labels (0 or 1) for each sample.

        Raises - TypeError
                Raised implicitly if ``fit`` has not been called (weights_ is None
                and np.dot will fail on a None operand).
        """
        z = self._linear_output(X)
        return self._step_activation(z)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the classification accuracy of the fitted model.
        Accuracy = (Number of correct predictions) / (Total predictions)

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns - float
                Fraction of correctly classified samples (range [0.0, 1.0]).
        """
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred))

    def plot_loss(self, title: str = "Perceptron Training Loss",
                  save_path: str = None) -> None:
        """
        Plot the MSE training loss curve across epochs.

        Parameters
        title : str, optional
            Title displayed above the plot.
        save_path : str or None, optional
            If provided, saves the figure to this file path (e.g.
            ``'loss_curve.png'``).  If None, calls ``plt.show()``.

        Returns None
        """
        if not self.loss_:
            raise RuntimeError("No loss data found. Call fit() before plot_loss().")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(self.loss_) + 1), self.loss_,
                color="#2c7bb6", linewidth=2, marker="o", markersize=3,
                markevery=max(1, len(self.loss_) // 20))
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(left=1)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Compute and return the confusion matrix as a labelled DataFrame.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns - pd.DataFrame
                Confusion matrix with rows labelled 'Actual' and columns
                labelled 'Predicted'.
        """
        y_pred = self.predict(X)
        return pd.crosstab(
            pd.Series(y, name="Actual"),
            pd.Series(y_pred, name="Predicted")
        )

    def get_params(self) -> dict:
        """
        Return the constructor hyperparameters of this Perceptron.

        Returns - dict
                Dictionary with keys ``learning_rate``, ``n_iterations``,
                and ``random_state``.
        """
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"Perceptron(learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"random_state={self.random_state})"
        )
