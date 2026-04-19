# Supervised Learing Models

```markdown
# supervised_learning

From-scratch implementations of supervised learning algorithms for both
classification and regression tasks. All estimators follow the consistent
`fit` / `predict` / `score` interface defined across the `rice_ml` package.

---

## Algorithms

### Linear Models

#### `linear_regression.py` — `LinearRegression`
Ordinary Least Squares (OLS) regression solved via gradient descent.
- **Parameters:** `learning_rate`, `n_iterations`, `random_state`
- **Attributes:** `weights_`, `bias_`, `loss_` (per-epoch MSE)
- **Methods:** `fit`, `predict`, `score` (R²), `plot_loss`
- **Evaluation:** MSE, RMSE, R²

#### `regresion.py` — `RidgeRegression`, `LassoRegression`
Regularised linear regression with L2 (Ridge) and L1 (Lasso) penalties.
Controls model complexity and prevents overfitting on high-dimensional data.
- **Key parameter:** `alpha` (regularisation strength)
- Ridge shrinks all coefficients smoothly toward zero.
- Lasso performs feature selection by driving sparse coefficients exactly to zero.

#### `logistic_regression.py` — `LogisticRegression`
Binary classification using sigmoid activation and cross-entropy loss,
optimised with batch gradient descent.
- **Parameters:** `learning_rate`, `n_iterations`, `random_state`
- **Attributes:** `weights_`, `bias_`, `loss_`
- **Methods:** `fit`, `predict`, `predict_proba`, `score` (accuracy)
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC

---

### Nearest Neighbours

#### `knn.py` — `KNNClassifier`, `KNNRegressor`
Distance-based K-Nearest Neighbours for classification and regression.
- **Parameters:** `n_neighbors`, `metric` (`'euclidean'` or `'manhattan'`)
- **Methods:** `fit`, `predict`, `score`
- Classification: majority vote among k nearest neighbours.
- Regression: mean of k nearest neighbour values.

---

### Decision Trees and Ensembles

#### `decision_tree_classifier.py` — `DecisionTreeClassifier`
Recursive CART decision tree for classification using Gini impurity
or information gain as the splitting criterion.
- **Parameters:** `max_depth`, `min_samples_split`, `criterion`
- **Methods:** `fit`, `predict`, `score`

#### `decision_tree_regressor.py` — `DecisionTreeRegressor`
Recursive CART decision tree for regression using variance reduction
as the splitting criterion.
- **Parameters:** `max_depth`, `min_samples_split`
- **Methods:** `fit`, `predict`, `score` (R²)

#### `ensemble.py` — `RandomForest`, `GradientBoosting`
- **RandomForest:** Bootstrap aggregation (bagging) of decision trees.
  Reduces variance. Includes feature importance scores.
  Parameters: `n_estimators`, `max_depth`, `max_features`, `random_state`
- **GradientBoosting:** Sequential boosting — each tree corrects the residuals
  of the previous. Reduces bias. Supports classification and regression.
  Parameters: `n_estimators`, `learning_rate`, `max_depth`, `random_state`

---

### Optimisation

#### `gradient_descent.py` — `GradientDescent`
General-purpose gradient descent optimiser used internally by linear models.
Supports batch, stochastic (SGD), and mini-batch modes.
- **Parameters:** `mode` (`'batch'`, `'sgd'`, `'mini-batch'`), `learning_rate`,
  `batch_size`, `n_iterations`

---

### Neural Networks

#### `perceptron.py` — `Perceptron`
Single-layer binary classifier with a Heaviside step activation function.
Tracks MSE loss per training epoch.
- **Parameters:** `learning_rate`, `n_iterations`, `random_state`
- **Attributes:** `weights_`, `bias_`, `loss_`, `n_features_in_`
- **Methods:** `fit`, `predict`, `accuracy`, `plot_loss`, `confusion_matrix`, `get_params`

#### `multi_layer_perceptron.py` — `MLP`
Feedforward neural network with configurable hidden layers, backpropagation,
dropout regularisation, and weight decay.
- **Parameters:** `hidden_layers` (list of layer sizes), `activation`, `learning_rate`,
  `n_iterations`, `dropout_rate`, `weight_decay`, `optimizer` (`'sgd'` or `'adam'`),
  `random_state`
- **Attributes:** `weights_`, `biases_`, `loss_`
- **Methods:** `fit`, `predict`, `predict_proba`, `score`, `plot_loss`

---

## Example Usage

```python
from rice_ml.supervised_learning import LogisticRegression
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

See the corresponding notebooks in [`examples/supervised_learning/`](../../../examples/supervised_learning/).
```

---
