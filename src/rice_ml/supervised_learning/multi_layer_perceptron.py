"""
multi_layer_perceptron.py

A simple implementation of a Multi-Layer Perceptron (MLP) for classification.

Builds a neural network from scratch using NumPy. The model can be
used for classification tasks where the goal is to assign each input example
to one of several possible classes.

The network works by:
- accepting input data as a matrix of samples and features
- passing the data through one or more hidden layers
- using sigmoid activation in the hidden layers
- using softmax activation in the output layer for class probabilities
- computing classification error with cross-entropy loss
- updating weights and biases using backpropagation and mini-batch gradient descent

This implementation is especially suited for learning how multi-layer
perceptrons work internally. For example, it can be used on flattened image
data such as MNIST digits, where each 28 by 28 image is reshaped into a
784-value input vector.
"""

import numpy as np


class multi_layer_perceptron:
    def __init__(
        self,
        layer_sizes,
        learning_rate=0.05,
        epochs=20,
        batch_size=32,
        random_state=42
    ):
        """
        Create a multi-layer perceptron classifier.

        Parameters
        ----------
        layer_sizes : list
            The number of neurons in each layer.

            Example:
            [784, 128, 64, 10]

            This means:
            - 784 input features
            - first hidden layer with 128 neurons
            - second hidden layer with 64 neurons
            - output layer with 10 neurons

        learning_rate : float
            Controls how large each weight update is during training.

        epochs : int
            Number of times the model passes through the full training dataset.

        batch_size : int
            Number of training examples used before each weight update.

        random_state : int
            Controls random initialization so results can be reproduced.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights = []
        self.biases = []
        self.loss_history = []
        self.classes_ = None

        self._initialize_parameters()

    # ---------------------------------------------------------
    # Activation functions
    # ---------------------------------------------------------

    def _sigmoid(self, z):
        """
        Sigmoid activation function.

        This is used for the hidden layers.
        It squashes values between 0 and 1.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, activated_value):
        """
        Derivative of the sigmoid function.

        The derivative is needed during backpropagation.
        """
        return activated_value * (1 - activated_value)

    def _softmax(self, z):
        """
        Softmax activation function.

        This is used in the output layer for multi-class classification.
        It converts raw output scores into probabilities.
        """
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(shifted_z)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # ---------------------------------------------------------
    # Parameter initialization
    # ---------------------------------------------------------

    def _initialize_parameters(self):
        """
        Initialize the weights and biases for each layer.

        Each pair of neighboring layers receives:
        - one weight matrix
        - one bias vector
        """
        rng = np.random.default_rng(self.random_state)

        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight_matrix = rng.normal(
                loc=0,
                scale=np.sqrt(2 / input_size),
                size=(input_size, output_size)
            )

            bias_vector = np.zeros((1, output_size))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    # ---------------------------------------------------------
    # Label encoding
    # ---------------------------------------------------------

    def _one_hot_encode(self, y):
        """
        Convert class labels into one-hot encoded vectors.

        Example:
        If the possible classes are [0, 1, 2], then:

        label 0 becomes [1, 0, 0]
        label 1 becomes [0, 1, 0]
        label 2 becomes [0, 0, 1]
        """
        self.classes_ = np.unique(y)

        class_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }

        y_indices = np.array([class_to_index[label] for label in y])

        one_hot = np.zeros((len(y), len(self.classes_)))
        one_hot[np.arange(len(y)), y_indices] = 1

        return one_hot

    # ---------------------------------------------------------
    # Feedforward
    # ---------------------------------------------------------

    def _forward(self, X):
        """
        Perform the feedforward step.

        The input data moves layer by layer through the network.
        Hidden layers use sigmoid activation.
        The output layer uses softmax activation.
        """
        activations = [X]
        weighted_sums = []

        current_activation = X

        for layer_index in range(len(self.weights)):
            z = current_activation @ self.weights[layer_index] + self.biases[layer_index]
            weighted_sums.append(z)

            output_layer = layer_index == len(self.weights) - 1

            if output_layer:
                current_activation = self._softmax(z)
            else:
                current_activation = self._sigmoid(z)

            activations.append(current_activation)

        return weighted_sums, activations

    # ---------------------------------------------------------
    # Loss function
    # ---------------------------------------------------------

    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Calculate cross-entropy loss.

        Cross-entropy measures how far the model's predicted probabilities
        are from the correct class labels.
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    # ---------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------

    def _backward(self, y_true, activations):
        """
        Perform backpropagation.

        Backpropagation calculates how much each weight and bias contributed
        to the model's error. These gradients are then used to update the model.
        """
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        number_of_examples = y_true.shape[0]

        # For softmax with cross-entropy loss, this is the output layer error.
        delta = activations[-1] - y_true

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = activations[layer_index]

            gradients_w[layer_index] = (
                previous_activation.T @ delta
            ) / number_of_examples

            gradients_b[layer_index] = (
                np.sum(delta, axis=0, keepdims=True)
            ) / number_of_examples

            if layer_index > 0:
                delta = (
                    delta @ self.weights[layer_index].T
                ) * self._sigmoid_derivative(activations[layer_index])

        return gradients_w, gradients_b

    # ---------------------------------------------------------
    # Updating weights and biases
    # ---------------------------------------------------------

    def _update_parameters(self, gradients_w, gradients_b):
        """
        Update weights and biases using gradient descent.
        """
        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= (
                self.learning_rate * gradients_w[layer_index]
            )

            self.biases[layer_index] -= (
                self.learning_rate * gradients_b[layer_index]
            )

    # ---------------------------------------------------------
    # Model training
    # ---------------------------------------------------------

    def fit(self, X, y):
        """
        Train the multi-layer perceptron.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (samples, features).")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        y_encoded = self._one_hot_encode(y)

        number_of_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            shuffled_indices = rng.permutation(number_of_samples)

            X_shuffled = X[shuffled_indices]
            y_shuffled = y_encoded[shuffled_indices]

            for start in range(0, number_of_samples, self.batch_size):
                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                _, activations = self._forward(X_batch)

                gradients_w, gradients_b = self._backward(
                    y_batch,
                    activations
                )

                self._update_parameters(gradients_w, gradients_b)

            _, full_activations = self._forward(X)
            epoch_loss = self._cross_entropy_loss(
                y_encoded,
                full_activations[-1]
            )

            self.loss_history.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

        return self

    # ---------------------------------------------------------
    # Prediction methods
    # ---------------------------------------------------------

    def predict_proba(self, X):
        """
        Return predicted class probabilities.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (samples, features).")

        _, activations = self._forward(X)

        return activations[-1]

    def predict(self, X):
        """
        Return the predicted class label for each input example.
        """
        probabilities = self.predict_proba(X)

        predicted_indices = np.argmax(probabilities, axis=1)

        return self.classes_[predicted_indices]

    def score(self, X, y):
        """
        Return the model's classification accuracy.
        """
        y = np.asarray(y)

        predictions = self.predict(X)

        return np.mean(predictions == y)