import numpy as np
from typing import Callable, List, Optional, Tuple


class GradientDescent:
    """
    Model Gradient Descent optimiser.

    This class implements only the optimisation loop.  All model-specific
    logic (cost function, gradient computation) is injected by the calling
    model at runtime, keeping this class reusable across any supervised
    learning model in the project (e.g. LinearRegression, LogisticRegression).

    Parameters
    ----------
    alpha : float, default=0.01
        Learning rate (step size).
    max_iter : int, default=1_000
        Maximum number of update steps (epochs).
    tol : float, optional
        Early-stopping tolerance.  Training halts when the absolute change
        in cost between consecutive iterations falls below `tol`.
    random_state : int, optional
        Seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_iter: int = 1_000,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Populated after calling optimize()
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.cost_history_: List[float] = []

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cost_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], float],
        gradient_func: Callable[
            [np.ndarray, np.ndarray, np.ndarray, float],
            Tuple[np.ndarray, float],
        ],
    ) -> Tuple[np.ndarray, float, List[float]]:
        r"""
        Run the gradient descent loop.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix, already validated and converted by the calling model.
        y : np.ndarray, shape (n_samples,)
            Target vector, already validated and converted by the calling model.
        cost_func : Callable[[X, y, weights, bias], float]
            Computes the scalar cost for the current parameters.
        gradient_func : Callable[[X, y, weights, bias], (dw, db)]
            Computes the parameter gradients for the current parameters.

        Returns
        -------
        weights : np.ndarray
            Optimised weight vector.
        bias : float
            Optimised bias term.
        cost_history : list of float
            Cost recorded at epoch 0 (before any update) and after every 
            subsequent update step.
        """
        n_samples, n_features = X.shape

        # --- Weight initialisation ---
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights_ = np.random.randn(n_features) * 0.01
        self.bias_ = float(np.random.randn() * 0.01)
        self.cost_history_ = []

        # Record cost before any update (epoch 0)
        self.cost_history_.append(cost_func(X, y, self.weights_, self.bias_))

        # --- Optimisation loop ---
        for _ in range(self.max_iter):

            # Compute gradients via the model-supplied function
            dw, db = gradient_func(X, y, self.weights_, self.bias_)

            # Parameter update
            self.weights_ -= self.alpha * dw
            self.bias_ -= self.alpha * db

            # Record cost after update
            current_cost = cost_func(X, y, self.weights_, self.bias_)
            self.cost_history_.append(current_cost)

            # Early stopping: compare current cost to the previous one
            if (
                self.tol is not None
                and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tol
            ):
                break

        return self.weights_, self.bias_, self.cost_history_


