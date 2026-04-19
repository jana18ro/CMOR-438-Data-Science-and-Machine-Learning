# Multi Layer Perceptron Notebook

```markdown
# Multi_Layer_Perceptron/

Full MLP training demonstration using `rice_ml.supervised_learning.MLP`,
including configurable architectures, regularisation, and optimiser comparison.

## Notebook

`mlp_notebook.ipynb`

## What the Notebook Covers

- **Architecture Design:** Choosing hidden layer sizes and depth;
  effect on model capacity and overfitting.
- **Forward Pass:** Layer-by-layer computation walkthrough with worked numerical example.
- **Backpropagation:** Gradient derivations and weight update mechanics.
- **Dropout Regularisation:** Comparing train/validation curves with and without dropout.
- **Weight Decay:** L2 penalty effect on learned weight distributions.
- **Optimiser Comparison:** SGD vs. Adam — convergence speed and final accuracy.
- **Evaluation:** Training/validation loss curves, accuracy, confusion matrix,
  per-class classification report.

## Dataset

`sklearn.datasets.load_digits` — 8×8 pixel handwritten digit images (10 classes).
64 features per sample; 1,797 samples total.

## Key imports

```python
from rice_ml.supervised_learning import MLP
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, confusion_matrix, classification_report
```
```

---
