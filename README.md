# 2026 CMOR 438: Data Science and Machine Learning 

---

# rice_ml: A Custom Machine Learning Library from Scratch

This repository hosts a custom-built machine learning package, **`rice_ml`**, developed as a core project for **CMOR 438 (Data Science and Machine Learning) at Rice University**. The project implements a comprehensive set of **supervised and unsupervised learning algorithms entirely from scratch** using Python and NumPy, alongside a rich collection of Jupyter notebooks covering classical ML, neural networks, dimensionality reduction, and more.

The primary goal is to move beyond black-box framework usage toward **algorithmic transparency** — emphasizing the mathematics, the mechanics, and the reproducibility of each model.

---

## Project Highlights

This repository serves as a complete, portfolio-ready codebase demonstrating:

- **Custom Core Implementations:** Full, from-scratch source code for all major ML algorithms, built on vectorized NumPy operations — no scikit-learn inside algorithm code.
- **Modular Package Design:** A cleanly organized, installable Python package (`rice_ml`) with a consistent `fit` / `predict` / `score` interface across all estimators.
- **Rigorous Preprocessing:** Dedicated modules for feature scaling, encoding, and data splitting, enforcing the "fit-on-train, transform-on-test" principle throughout.
- **Educational Jupyter Notebooks:** Structured example notebooks in `examples/` covering data exploration, preprocessing, modeling, evaluation, and visualization for every algorithm.
- **Quality Assurance:** A comprehensive `pytest` test suite with unit tests for all models, utilities, and edge cases.
- **Automated CI/CD:** GitHub Actions workflow runs the full test suite on every push and pull request.

---

## Repository Structure

```bash
2026_Data_Science_and_Machine_Learning/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── pull_request_template.md
├── examples/
│   ├── supervised_learning/
│   └── unsupervised_learning/
├── src/
│   └── rice_ml/
│       ├── supervised_learning/
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── knn.py
│       │   ├── decision_tree_classifier.py
│       │   ├── decision_tree_regressor.py
│       │   ├── ensemble.py
│       │   ├── gradient_descent.py
│       │   ├── perceptron.py
│       │   └── multi_layer_perceptron.py
│       ├── unsupervised_learning/
│       │   ├── kmeans.py
│       │   ├── dbscan.py
│       │   ├── pca.py
│       │   └── label_propagation_community_detection.py
│       ├── processing/
│       ├── measures/
├── tests/
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_knn.py
│   ├── test_decision_tree_classifier.py
│   ├── test_decision_tree_regressor.py
│   ├── test_ensemble.py
│   ├── test_kmeans.py
│   ├── test_dbscan.py
│   ├── test_pca.py
│   ├── test_label_propagation_community_detection.py
│   ├── test_perceptron.py
│   ├── test_multi_layer_perceptron.py
│   ├── test_metrics.py
│   ├── test_preprocess.py
│   └── test_postprocess.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
└── pytest.ini
```

---

## Algorithms Implemented

### Supervised Learning

Implemented in [`src/rice_ml/supervised_learning/`](src/rice_ml/supervised_learning/)

| Algorithm | File | Description |
|-----------|------|-------------|
| **Linear Regression** | `linear_regression.py` | Ordinary Least Squares (OLS) via gradient descent; RMSE and R² evaluation |
| **Ridge / Lasso Regression** | `regresion.py` | L2 and L1 regularised regression with coefficient path analysis |
| **Logistic Regression** | `logistic_regression.py` | Binary classification with sigmoid activation, cross-entropy loss, batch gradient descent |
| **K-Nearest Neighbours** | `knn.py` | Distance-based classifier and regressor; supports Euclidean and Manhattan metrics |
| **Decision Tree Classifier** | `decision_tree_classifier.py` | Recursive CART tree using Gini impurity and information gain |
| **Decision Tree Regressor** | `decision_tree_regressor.py` | Recursive CART tree using variance reduction as the splitting criterion |
| **Ensemble Methods** | `ensemble.py` | Random Forest (bagging) and Gradient Boosting (sequential boosting) |
| **Gradient Descent** | `gradient_descent.py` | General gradient descent optimiser (batch, stochastic, mini-batch) |
| **Perceptron** | `perceptron.py` | Single-layer binary classifier with step activation; MSE loss tracking |
| **Multi-Layer Perceptron** | `multi_layer_perceptron.py` | Feedforward neural network with backpropagation, dropout, and weight decay |

### Unsupervised Learning

Implemented in [`src/rice_ml/unsupervised_learning/`](src/rice_ml/unsupervised_learning/)

| Algorithm | File | Description |
|-----------|------|-------------|
| **K-Means** | `kmeans.py` | Centroid-based clustering with inertia and elbow-method support |
| **DBSCAN** | `dbscan.py` | Density-based clustering for arbitrarily shaped clusters and noise detection |
| **PCA** | `pca.py` | Principal Component Analysis via eigen-decomposition; explained variance analysis |
| **Label Propagation** | `label_propagation_community_detection.py` | Graph-based community detection via iterative neighbourhood label spreading |

### Data Processing & Evaluation

| Module | File | Description |
|--------|------|-------------|
| **Preprocessing** | `processing/preprocess.py` | `StandardScaler`, `MinMaxScaler`, `OrdinalEncoder`, `train_test_split` |
| **Postprocessing** | `processing/postprocess.py` | Probability thresholding, label decoding, output formatting |
| **Metrics** | `measures/metrics.py` | `accuracy_score`, `precision`, `recall`, `f1_score`, `mse`, `rmse`, `r2_score`, confusion matrix |
| **Validation** | `measures/validation.py` | K-fold cross-validation, stratified split, holdout evaluation |

---

## Example Notebooks

All example notebooks follow a consistent workflow: **load data → explore → preprocess → train → evaluate → visualise**.

### Supervised Learning Examples

| Notebook | Location | Dataset |
|----------|----------|---------|
| Linear & Ridge & Lasso Regression | [`examples/supervised_learning/Linear_Regression/`](examples/supervised_learning/Linear_Regression/) | housing.csv |
| Logistic Regression | [`examples/supervised_learning/Logistic_Regression/`](examples/supervised_learning/Logistic_Regression/) | cancer_data.csv |
| K-Nearest Neighbours | [`examples/supervised_learning/KNN/`](examples/supervised_learning/KNN/) | iris.csv |
| Decision Tree Classifier | [`examples/supervised_learning/Decision_Trees/`](examples/supervised_learning/Decision_Trees/) | iris.csv |
| Regression Tree | [`examples/supervised_learning/Regression_Trees/`](examples/supervised_learning/Regression_Trees/) | iris.csv |
| Ensemble Methods | [`examples/supervised_learning/Ensembles/`](examples/supervised_learning/Ensembles/) | titanic.csv |
| Perceptron — Fake News | [`examples/supervised_learning/Perceptron/`](examples/supervised_learning/Perceptron/) | Fake.csv / True.csv |
| Neural Networks | [`examples/supervised_learning/Neural_Networks/`](examples/supervised_learning/Neural_Networks/) | MNIST.csv |


### Unsupervised Learning Examples

| Notebook | Location | Dataset |
|----------|----------|---------|
| K-Means Clustering | [`examples/unsupervised_learning/K-means/`](examples/unsupervised_learning/K-means/) | mall_customers.csv |
| DBSCAN | [`examples/unsupervised_learning/DBSCAN/`](examples/unsupervised_learning/DBSCAN/) | `make_moons`, `make_circles` |
| PCA | [`examples/unsupervised_learning/PCA/`](examples/unsupervised_learning/PCA/) | iris.csv |
| Label Propagation / Community Detection | [`examples/unsupervised_learning/Label_Propagation_Community_Detection/`](examples/unsupervised_learning/Label_Propagation_Community_Detection/) | Synthetic datasets |

---

## Installation

This package is not published to PyPI. Install locally in editable mode:

```bash
git clone https://github.com/<your-username>/2026_Data_Science_and_Machine_Learning.git
cd 2026_Data_Science_and_Machine_Learning
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

---

## Getting Started

```python
from rice_ml.supervised_learning import Perceptron
from rice_ml.processing import train_test_split, StandardScaler
from rice_ml.measures import accuracy_score
import numpy as np

# Example: train Perceptron on synthetic data
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [2, 2], [2, 0]])
y = np.array([0, 1, 1, 0, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

clf = Perceptron(learning_rate=0.1, n_iterations=200, random_state=42)
clf.fit(X_train_scaled, y_train)

print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_scaled)))
clf.plot_loss()
```

---

## Testing

The project uses `pytest` with a comprehensive unit test suite covering all algorithms, preprocessing utilities, and evaluation metrics.

Run the full test suite from the repository root:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run with coverage report:

```bash
pytest --cov=rice_ml --cov-report=term-missing
```

Tests run automatically on every push and pull request via [GitHub Actions](.github/workflows/test.yml).

---

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn (for datasets and metrics comparison only — not used inside algorithm implementations)
- pytest
- jupyter

See [`requirements.txt`](requirements.txt) for the full pinned dependency list.

---

## Author and License

**Author:** Jana — CMOR 438, Data Science and Machine Learning, Rice University, 2026  
**License:** MIT License — see [`LICENSE`](LICENSE) for details.
