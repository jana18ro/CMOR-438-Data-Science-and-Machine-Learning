# Examples: Machine Learning From Scratch

# examples/

This directory contains Jupyter notebooks demonstrating the `rice_ml` package
on real and synthetic datasets. Each notebook covers a complete ML workflow:

> **load data → explore → preprocess → train → evaluate → visualise**

Notebooks are organized into two top-level categories matching the package structure.

---

## Structure

``` bash
examples/
├── supervised_learning/
│   ├── Linear_Regression/
│   ├── Logistic_Regression/
│   ├── KNN/
│   ├── Decision_Trees/
│   ├── Regression_Trees/
│   ├── Ensembles/
│   ├── Perceptron/
│   │   ├── perceptron_notebook.ipynb
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── Neural_Networks/
│   └── Multi_Layer_Perceptron/
└── unsupervised_learning/
    ├── K-means/
    ├── DBSCAN/
    ├── PCA/
    ├── SVD/
    └── Label_Propagation_Community_Detection/
```

---

## How to Run

1. **Install the package** from the project root:

   ```bash
   pip install -e ..
   ```

2. **Install Jupyter** if not already installed:

   ```bash
   pip install jupyter
   ```

3. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

4. Navigate to any notebook folder and open the `.ipynb` file.

---

## Supervised Learning Notebooks

| Folder | Algorithm(s) | Dataset(s) |
|--------|-------------|------------|
| [`Linear_Regression/`](supervised_learning/Linear_Regression/) | OLS, Ridge, Lasso | housing |
| [`Logistic_Regression/`](supervised_learning/Logistic_Regression/) | Logistic Regression | breast cancer |
| [`KNN/`](supervised_learning/KNN/) | KNN Classifier & Regressor | Iris |
| [`Decision_Trees/`](supervised_learning/Decision_Trees/) | CART Classifier | Iris |
| [`Ensembles/`](supervised_learning/Ensembles/) | Random Forest, Gradient Boosting | Titanic |
| [`Perceptron/`](supervised_learning/Perceptron/) | Single-Layer Perceptron | Fake/Real News |
| [`Neural_Networks/`](supervised_learning/Neural_Networks/) | Backpropagation, Activations, CNNs | MNIST-style |

## Unsupervised Learning Notebooks

| Folder | Algorithm(s) | Dataset(s) |
|--------|-------------|------------|
| [`K-means/`](unsupervised_learning/K-means/) | K-Means Clustering | mall customers |
| [`DBSCAN/`](unsupervised_learning/DBSCAN/) | Density-Based Clustering | `make_moons`, `make_circles` |
| [`PCA/`](unsupervised_learning/PCA/) | Principal Component Analysis | Iris (4D → 2D) |
| [`Label_Propagation_Community_Detection/`](unsupervised_learning/Label_Propagation_Community_Detection/) | Community Detection | Synthetic |
```
