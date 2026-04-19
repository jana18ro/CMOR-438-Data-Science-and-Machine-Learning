# Processing Utilities

```markdown
# processing

Data preprocessing and postprocessing utilities for the `rice_ml` pipeline.

A core principle throughout this package: **scalers and encoders are always
fit exclusively on training data** and then applied (transform-only) to
validation and test sets. This prevents data leakage and ensures honest
model evaluation.

---

## `preprocess.py`

### `StandardScaler`
Zero-mean, unit-variance standardisation.
Computes `mean_` and `std_` from the training set only.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled  = scaler.transform(X_test)        # transform only (no refit)
```

### `MinMaxScaler`
Scales each feature to the range [0, 1] (or a custom `feature_range`).
Computes `min_` and `scale_` from the training set only.

```python
mm = MinMaxScaler(feature_range=(0, 1))
X_train_norm = mm.fit_transform(X_train)
X_test_norm  = mm.transform(X_test)
```

### `OrdinalEncoder`
Converts categorical string features to integer codes.
Stores `categories_` from the training set; unseen categories at test time
are handled gracefully (configurable `handle_unknown` strategy).

```python
enc = OrdinalEncoder()
X_train_enc = enc.fit_transform(X_train_categorical)
X_test_enc  = enc.transform(X_test_categorical)
```

### `train_test_split(X, y, test_size, random_state)`
Splits feature and label arrays into random train and test subsets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## `postprocess.py`

### `threshold_predictions(probs, threshold=0.5)`
Converts continuous probability outputs from a classifier into binary labels
using a configurable decision threshold.

```python
y_pred = threshold_predictions(model.predict_proba(X_test), threshold=0.4)
```

### `decode_labels(encoded, mapping)`
Reverses an integer encoding back to the original category strings using
a provided dictionary mapping.

### `format_predictions(y_pred, index)`
Wraps a prediction array in a `pandas.Series` with a provided index for
clean tabular output and downstream compatibility.
```

---