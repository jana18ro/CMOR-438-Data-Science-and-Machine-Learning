# FILE: 2026_Data_Science_and_Machine_Learning\examples\supervised_learning\README.md

```markdown
# examples/supervised_learning/

Jupyter notebooks demonstrating the supervised learning algorithms from
`rice_ml.supervised_learning` on real and synthetic datasets.

Every notebook follows the same structured workflow:
1. **Data Loading** — load dataset, inspect shape and dtypes
2. **Exploratory Data Analysis** — distributions, correlations, class balance
3. **Preprocessing** — scaling, encoding, train/test split (using `rice_ml.processing`)
4. **Model Training** — fit the `rice_ml` estimator
5. **Evaluation** — metrics, loss curves, decision boundaries (using `rice_ml.measures`)
6. **Visualisation** — plots illustrating model behaviour and results

---

## Contents

| Folder | Algorithm | Key Concepts | Dataset |
|--------|-----------|-------------|---------|
| [`Linear_Regression/`](Linear_Regression/) | OLS, Ridge, Lasso | Gradient descent, regularisation, coefficient paths | Synthetic / housing |
| [`Logistic_Regression/`](Logistic_Regression/) | Logistic Regression | Sigmoid, cross-entropy, decision boundary | Iris, breast cancer |
| [`KNN/`](KNN/) | KNN Classifier & Regressor | Distance metrics, k vs. accuracy trade-off | Iris |
| [`Decision_Trees/`](Decision_Trees/) | CART Classifier | Gini, information gain, tree visualisation | Titanic, Iris |
| [`Regression_Trees/`](Regression_Trees/) | CART Regressor | Variance reduction, depth vs. smoothness | Synthetic |
| [`Ensembles/`](Ensembles/) | Random Forest, Gradient Boosting | Bagging, boosting, feature importance | Titanic, housing |
| [`Perceptron/`](Perceptron/) | Perceptron | Step activation, TF-IDF, MSE loss curve | Fake/Real News CSVs |
| [`Neural_Networks/`](Neural_Networks/) | Backprop, Activations, CNNs | SGD vs Adam, dropout, weight decay | XOR / MNIST-style |
| [`Multi_Layer_Perceptron/`](Multi_Layer_Perceptron/) | MLP | Hidden layers, regularisation, loss curves | `load_digits` |
```

---
