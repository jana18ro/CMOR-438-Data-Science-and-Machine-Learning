# Test Suite

```markdown
# tests/

This directory contains the full unit test suite for the `rice_ml` package,
written using `pytest`. Every algorithm, utility, and metric in the library
has dedicated tests covering correct outputs, consistent interface behaviour,
and meaningful edge cases.

---

## Running the Tests

From the repository root:

```bash
# Run all tests
pytest

# Verbose output (shows individual test names)
pytest -v

# Run a single test file
pytest tests/test_perceptron.py

# Run tests matching a keyword
pytest -k "perceptron or knn"

# Run with coverage report
pytest --cov=rice_ml --cov-report=term-missing
```

Tests also run automatically on every push to `main` and on every
pull request via [GitHub Actions](../.github/workflows/test.yml).

---

## Test Files

| File | Covers |
|------|--------|
| `test_linear_regression.py` | Weight convergence, MSE/R² values, gradient updates |
| `test_logistic_regression.py` | Binary classification accuracy, loss convergence, `predict_proba` |
| `test_knn.py` | `KNNClassifier` majority vote, `KNNRegressor` mean output, distance metrics |
| `test_decision_tree_classifier.py` | Gini splits, max_depth constraints, correct leaf labels |
| `test_decision_tree_regressor.py` | Variance reduction splits, prediction on seen/unseen data |
| `test_ensemble.py` | `RandomForest` majority vote, `GradientBoosting` residual correction |
| `test_kmeans.py` | Centroid convergence, label assignment, inertia monotonicity |
| `test_dbscan.py` | Cluster labels, noise detection (`-1`), core sample identification |
| `test_pca.py` | Explained variance ratios sum ≤ 1, correct `n_components` in output |
| `test_svd.py` | Reconstruction accuracy at various ranks, shape correctness |
| `test_label_propagation_community_detection.py` | Label stability at convergence, community count |
| `test_perceptron.py` | Weight updates, step activation, `plot_loss`, `confusion_matrix` |
| `test_multi_layer_perceptron.py` | Forward pass shapes, loss decrease over epochs, dropout masking |
| `test_metrics.py` | `accuracy_score`, `f1_score`, `r2_score` against known ground-truth values |
| `test_preprocess.py` | Scaler fit/transform, no-leakage on test set, `train_test_split` proportions |
| `test_postprocess.py` | `threshold_predictions` at various thresholds, `decode_labels` mapping |

---

## Testing Philosophy

- **Small synthetic datasets:** All tests use tiny arrays generated with NumPy so
  the suite runs in seconds and does not depend on external data files.
- **Known expected values:** Where possible, expected outputs are computed by hand
  or verified against scikit-learn as a reference implementation.
- **Edge cases:** Tests include empty arrays, single-sample inputs, k=1 for KNN,
  max_depth=1 for trees, and n_clusters > n_samples for K-Means.
- **Reproducibility:** All stochastic tests pass `random_state=42` to ensure
  deterministic results across platforms.

---

## Adding New Tests

1. Create a new file `tests/test_<module>.py`.
2. Import the estimator or function under test from `rice_ml`.
3. Write test functions prefixed with `test_`.
4. Use small synthetic data — avoid loading real datasets in unit tests.
5. Run `pytest -v` to verify your tests are discovered and passing.
```

---
