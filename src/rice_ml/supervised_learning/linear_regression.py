"""
linear_regression.py
--------------------
Simple Linear Regression via Gradient Descent (Regression Neuron).

The model learns a single weight and bias by minimising Mean Squared Error:

    y_hat = w * x + b
    L     = (1 / n) * sum((y - y_hat) ** 2)

All optimisation is delegated to GradientDescent, keeping this file
responsible only for the MSE cost / gradient logic and the public API.

Mini example
------------
>>> import numpy as np
>>> from linear_regression import SimpleLinearRegression
>>> X = np.array([1., 2., 3., 4., 5.])
>>> y = np.array([2., 4., 6., 8., 10.])           # y = 2x
>>> model = SimpleLinearRegression(alpha=0.01, max_iter=2000)
>>> model = model.fit(X, y)
>>> round(float(model.coef_), 4)
2.0
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from .gradient_descent import GradientDescent

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _to_column(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Coerce a 1-D array-like into a float column vector of shape (n, 1).

    GradientDescent.optimize() expects X of shape (n_samples, n_features),
    so for a single-feature model we always pass a (n, 1) matrix.

    Parameters
    ----------
    X    : 1-D array-like of input values.
    name : Variable name used in error messages.

    Returns
    -------
    np.ndarray  shape (n_samples, 1), dtype float.
    """
    arr = np.asarray(X, dtype=float)

    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr                          # already (n, 1)

    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (a single feature vector); got shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr.reshape(-1, 1)


def _to_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Coerce target values into a 1-D float array.

    Parameters
    ----------
    y    : 1-D array-like of target values.
    name : Variable name used in error messages.

    Returns
    -------
    np.ndarray  shape (n_samples,), dtype float.
    """
    arr = np.asarray(y, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


# ---------------------------------------------------------------------------
# MSE cost and gradient (injected into GradientDescent)
# ---------------------------------------------------------------------------

def _mse_cost(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> float:
    """
    Mean Squared Error loss.

        L = (1 / n) * sum((y - (w * x + b)) ** 2)

    Parameters
    ----------
    X       : shape (n, 1)  — feature column.
    y       : shape (n,)    — targets.
    weights : shape (1,)    — current weight [w].
    bias    : float         — current bias b.

    Returns
    -------
    float  scalar MSE value.
    """
    y_hat = (X @ weights) + bias        # (n, 1) @ (1,) + scalar -> (n,)
    residuals = y - y_hat.ravel()
    return float(np.mean(residuals ** 2))


def _mse_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
):
    """
    Gradients of MSE w.r.t. the weight vector and bias.

        dL/dw = (-2 / n) * X^T (y - y_hat)
        dL/db = (-2 / n) * sum(y - y_hat)

    Parameters
    ----------
    X       : shape (n, 1).
    y       : shape (n,).
    weights : shape (1,)  — current weight [w].
    bias    : float       — current bias.

    Returns
    -------
    dw : np.ndarray  shape (1,)  — gradient w.r.t. w.
    db : float                   — gradient w.r.t. b.
    """
    n = len(y)
    y_hat = (X @ weights) + bias
    error = y - y_hat.ravel()           # (n,)

    dw = (-2 / n) * (X.T @ error)      # (1, n) @ (n,) -> (1,)
    db = float((-2 / n) * np.sum(error))

    return dw, db


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleLinearRegression:
    """
    Simple (single-feature) Linear Regression via Gradient Descent.

    Delegates all parameter updates to :class:`GradientDescent`, which
    handles the optimisation loop, early stopping, and cost history.

    Model
    -----
        y_hat = coef_ * x + intercept_

    Parameters
    ----------
    alpha        : float, default=0.01
        Learning rate passed to GradientDescent.
    max_iter     : int, default=1_000
        Maximum number of gradient descent steps.
    tol          : float, optional
        Early-stopping tolerance.  Passed straight through to GradientDescent;
        training stops when |cost[t-1] - cost[t]| < tol.
    random_state : int, optional
        Seed for reproducible weight initialisation inside GradientDescent.

    Attributes
    ----------
    coef_         : float | None
        Learned slope (w).  Set after :meth:`fit`.
    intercept_    : float | None
        Learned intercept (b).  Set after :meth:`fit`.
    cost_history_ : list of float
        MSE recorded at epoch 0 (before any update) and after every subsequent
        step.  Populated by GradientDescent.optimize().

    Examples
    --------
    >>> import numpy as np
    >>> from linear_regression import SimpleLinearRegression
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> y = np.array([3., 5., 7., 9., 11.])          # y = 2x + 1
    >>> model = SimpleLinearRegression(alpha=0.01, max_iter=3000).fit(X, y)
    >>> round(float(model.coef_), 1), round(float(model.intercept_), 1)
    (2.0, 1.0)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_iter: int = 1_000,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha (learning rate) must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.coef_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.cost_history_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SimpleLinearRegression":
        """
        Train the model on (X, y).

        Parameters
        ----------
        X : 1-D array-like of shape (n_samples,)
            Single input feature.
        y : 1-D array-like of shape (n_samples,)
            Regression targets.

        Returns
        -------
        self : SimpleLinearRegression
            Fitted model (enables method chaining).
        """
        X_col = _to_column(X, "X")     # (n, 1)  — shape GradientDescent expects
        y_arr = _to_1d(y, "y")

        if X_col.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got {X_col.shape[0]} vs {y_arr.shape[0]}."
            )

        # Build the optimizer and run the loop
        optimizer = GradientDescent(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        weights, bias, cost_history = optimizer.optimize(
            X_col,
            y_arr,
            cost_func=_mse_cost,
            gradient_func=_mse_gradients,
        )

        # GradientDescent returns weights as shape (1,); unpack to a scalar
        self.coef_ = float(weights[0])
        self.intercept_ = float(bias)
        self.cost_history_ = cost_history

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predictions for new inputs.

        Parameters
        ----------
        X : 1-D array-like of shape (n_samples,)

        Returns
        -------
        np.ndarray  shape (n_samples,)  predicted values.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted — call fit() first.")

        X_arr = _to_1d(X, "X")
        return self.coef_ * X_arr + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R² coefficient of determination.

        Returns 1.0 for a perfect fit, 0.0 when the model only predicts the
        mean, and negative values when the fit is worse than predicting the mean.

        Parameters
        ----------
        X : 1-D array-like
        y : 1-D array-like

        Returns
        -------
        float  R² value.
        """
        y_arr = _to_1d(y, "y")
        y_hat = self.predict(X)

        ss_res = np.sum((y_arr - y_hat) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1.0 - ss_res / ss_tot)

    def plot_cost(self) -> None:
        """Plot the MSE cost curve recorded during training."""
        if not self.cost_history_:
            raise RuntimeError("No cost history — call fit() first.")

        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_history_, color="steelblue", linewidth=2)
        plt.title("Training Cost (MSE) over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Scatter-plot the data and overlay the fitted regression line.

        Parameters
        ----------
        X : 1-D array-like  input feature values.
        y : 1-D array-like  target values.
        """
        X_arr = _to_1d(X, "X")
        y_arr = _to_1d(y, "y")
        y_hat = self.predict(X_arr)

        plt.figure(figsize=(8, 5))
        plt.scatter(X_arr, y_arr, color="steelblue", label="Data", zorder=3)
        plt.plot(
            X_arr, y_hat, color="tomato", linewidth=2,
            label=f"Fit: y = {self.coef_:.4f}x + {self.intercept_:.4f}",
        )
        plt.title("Simple Linear Regression — Fitted Line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        if self.coef_ is None:
            return "SimpleLinearRegression (unfitted)"
        return (
            f"SimpleLinearRegression(\n"
            f"  alpha={self.alpha}, max_iter={self.max_iter}\n"
            f"  coef_      (w) = {self.coef_:.6f}\n"
            f"  intercept_ (b) = {self.intercept_:.6f}\n"
            f")"
        )


