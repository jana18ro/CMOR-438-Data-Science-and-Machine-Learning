# FILE: 2026_Data_Science_and_Machine_Learning\examples\README.md

```markdown
# examples/

This directory contains Jupyter notebooks demonstrating the `rice_ml` package
on real and synthetic datasets. Each notebook covers a complete ML workflow:

> **load data → explore → preprocess → train → evaluate → visualise**

Notebooks are organized into two top-level categories matching the package structure.

---

## Structure

```
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
| [`Linear_Regression/`](supervised_learning/Linear_Regression/) | OLS, Ridge, Lasso | Synthetic / housing |
| [`Logistic_Regression/`](supervised_learning/Logistic_Regression/) | Logistic Regression | Iris, breast cancer |
| [`KNN/`](supervised_learning/KNN/) | KNN Classifier & Regressor | Iris, synthetic |
| [`Decision_Trees/`](supervised_learning/Decision_Trees/) | CART Classifier | Titanic, Iris |
| [`Regression_Trees/`](supervised_learning/Regression_Trees/) | CART Regressor | Synthetic non-linear |
| [`Ensembles/`](supervised_learning/Ensembles/) | Random Forest, Gradient Boosting | Titanic, housing |
| [`Perceptron/`](supervised_learning/Perceptron/) | Single-Layer Perceptron | Fake/Real News |
| [`Neural_Networks/`](supervised_learning/Neural_Networks/) | Backpropagation, Activations, CNNs | XOR, MNIST-style |
| [`Multi_Layer_Perceptron/`](supervised_learning/Multi_Layer_Perceptron/) | MLP with Dropout | `load_digits` |

## Unsupervised Learning Notebooks

| Folder | Algorithm(s) | Dataset(s) |
|--------|-------------|------------|
| [`K-means/`](unsupervised_learning/K-means/) | K-Means Clustering | Iris, `make_blobs` |
| [`DBSCAN/`](unsupervised_learning/DBSCAN/) | Density-Based Clustering | `make_moons`, `make_circles` |
| [`PCA/`](unsupervised_learning/PCA/) | Principal Component Analysis | Iris (4D → 2D) |
| [`SVD/`](unsupervised_learning/SVD/) | Singular Value Decomposition | Grayscale image |
| [`Label_Propagation_Community_Detection/`](unsupervised_learning/Label_Propagation_Community_Detection/) | Community Detection | Karate Club graph |
```

---
