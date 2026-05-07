# Source Folder


# src/

This directory contains the full source code for the `rice_ml` package — a custom,
from-scratch machine learning library built for CMOR 438 at Rice University.

## Structure

```
src/
└── rice_ml/
    ├── supervised_learning/      # Classifiers and regressors
    ├── unsupervised_learning/    # Clustering and dimensionality reduction
    ├── processing/               # Preprocessing and postprocessing utilities
    ├── measures/                 # Evaluation metrics and cross-validation
    └── __init__.py
```

## Installing the Package

Install in **editable mode** from the project root so that any changes to source files
are immediately reflected without reinstalling:

```bash
pip install -e .
```

Once installed, import directly:

```python
from rice_ml.supervised_learning import Perceptron, LinearRegression
from rice_ml.unsupervised_learning import KMeans, PCA
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, r2_score
```

## Design Notes

- All algorithm files in this package are **pure NumPy implementations**.
  scikit-learn is used only in example notebooks for dataset loading and
  external metric comparisons — never inside the algorithm source code itself.
- Every estimator follows a consistent `fit` / `predict` / `score` interface.
- `random_state` parameters are used throughout for full reproducibility.
- See the [`rice_ml/` README](rice_ml/README.md) for a full capability overview.
```
