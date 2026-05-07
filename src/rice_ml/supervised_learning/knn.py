from __future__ import annotations
import numpy as np

def distance(a, b):
    """
    Compute the Euclidean distance between two vectors.
    
    Parameters
    ----------
    a : array_like
        First input vector.
    b : array_like
        Second input vector.
        
    Returns
    -------
    float
        The L2 distance between a and b.
    """
    return np.sqrt(np.sum((a - b)**2))

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    This class implements a simple version of the KNN algorithm. It is a 
    lazy learner, meaning it does not build a model during the fitting phase; 
    instead, it stores the entire training dataset and performs computations 
    during the prediction phase.

    Parameters
    ----------
    k : int
        The number of nearest neighbors to consider for classification.

    Attributes
    ----------
    k : int
        Number of neighbors.
    X_train : np.ndarray
        Stored training feature matrix.
    y_train : np.ndarray
        Stored training labels.

    Notes
    -----
    - Prediction is performed by finding the k samples in the training set 
      closest to the query point and taking a majority vote of their labels.
    - Ties are broken based on the order of labels in the Python set 
      transformation, preserving the behavior described in the guide.
    """

    def __init__(self, k: int):
        """
        Initialize the KNN classifier with a specific k value.

        Parameters
        ----------
        k : int
            Number of neighbors to use for voting.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Store the training data for later use during prediction.

        Parameters
        ----------
        X_train : array_like
            Training feature matrix of shape (n_samples, n_features).
        y_train : array_like
            Target labels of shape (n_samples,).
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict the labels for the provided test data.

        Parameters
        ----------
        X_test : array_like
            Test feature matrix of shape (n_test_samples, n_features).

        Returns
        -------
        list
            A list of predicted labels for each sample in X_test.
        """
        predictions = []

        for query_point in X_test:
            # Get the k-nearest neighbors for the current point
            neighbors = self._k_nearest(query_point)

            # Extract labels from the neighbor tuples (x_train, label, distance)
            votes = [label for _, label, _ in neighbors]

            # Determine the majority label
            # Ties are resolved by the default behavior of max() on set()
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)

        return predictions

    def _k_nearest(self, point):
        """
        Find the k training samples closest to a single query point.

        Parameters
        ----------
        point : array_like
            The single query point to classify.

        Returns
        -------
        list of tuples
            The k closest neighbors, each represented as (x_train, label, distance),
            sorted in ascending order of distance.
        """
        distances = []

        # Iterate through all stored training samples
        for x_train, label in zip(self.X_train, self.y_train): # type: ignore
            d = distance(point, x_train)
            distances.append((x_train, label, d))

        # Sort the list based on the distance (the 3rd element in the tuple)
        distances.sort(key=lambda x: x[2])

        # Return only the first k neighbors
        return distances[:self.k]

# -----------------------------------------------------------------------------
# End of KNN module
# -----------------------------------------------------------------------------