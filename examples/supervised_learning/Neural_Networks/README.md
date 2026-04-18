# FILE: 2026_Data_Science_and_Machine_Learning\examples\supervised_learning\Neural_Networks\README.md

```markdown
# Neural_Networks/

Introduction to feed-forward neural networks — activation functions,
backpropagation, optimisation, convolutional neural networks (CNNs), and
regularisation techniques.

## Notebook

`neural_networks_notebook.ipynb`

## What the Notebook Covers

- **Activation Functions:** Sigmoid, ReLU, tanh, softmax — formulas, plots,
  and vanishing gradient discussion.
- **Backpropagation:** Step-by-step derivation of gradient computation through
  a two-layer network; chain rule worked example.
- **Optimisers:** Side-by-side comparison of SGD, momentum SGD, and Adam
  on a simple non-convex loss surface.
- **Introductory CNNs:** Convolution operation, pooling, receptive fields;
  brief classification demo on image patches.
- **Regularisation:**
  - **Dropout:** Masking neurons at training time; effect on generalisation.
  - **Weight Decay (L2):** Penalising large weights; connection to Ridge regression.
- **Visualisations:** Loss and accuracy curves, weight distribution histograms,
  feature map activations.

## Dataset

- XOR problem (illustrates need for non-linear activations).
- MNIST-style digit patches (for CNN introductory demo).

## Key imports

```python
from rice_ml.supervised_learning import MLP
from rice_ml.processing import StandardScaler, train_test_split
from rice_ml.measures import accuracy_score, cross_val_score
```
```

---