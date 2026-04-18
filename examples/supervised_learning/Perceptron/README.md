# FILE: 2026_Data_Science_and_Machine_Learning\examples\supervised_learning\Perceptron\README.md

```markdown
# Perceptron/

Demonstrates `rice_ml.supervised_learning.Perceptron` on a real-world
binary text classification task: distinguishing fake from genuine news articles.

## Notebook

`perceptron_notebook.ipynb`

## Dataset

| File | Description | Size |
|------|-------------|------|
| `Fake.csv` | Fabricated / fake news articles with title, text, subject, date | ~23,000 articles |
| `True.csv` | Verified / real news articles with title, text, subject, date | ~21,000 articles |

Both files must be present in this directory before running the notebook.

## What the Notebook Covers

1. **Data Loading & Merging** — combine `Fake.csv` and `True.csv` with binary labels (0 = Fake, 1 = Real).
2. **Exploratory Data Analysis** — article length distributions, subject breakdown, class balance.
3. **Text Preprocessing** — lowercasing, punctuation removal, stopword filtering.
4. **Feature Engineering** — TF-IDF vectorisation of article text; dimensionality discussed.
5. **Perceptron Training** — fit `rice_ml.Perceptron` on TF-IDF features.
6. **Loss Curve** — MSE per epoch via `clf.plot_loss()`; convergence analysis.
7. **Evaluation** — accuracy, confusion matrix (`clf.confusion_matrix()`), precision, recall, F1.
8. **Discussion** — limitations of the single-layer perceptron; comparison with MLP.

## Key imports

```python
from rice_ml.supervised_learning import Perceptron
from rice_ml.processing import train_test_split
from rice_ml.measures import accuracy_score, confusion_matrix, classification_report
```

## Notes

- The dataset is large (~116 MB combined). TF-IDF vocabulary is capped to control
  feature dimensionality — see notebook for the chosen `max_features` setting.
- Training is slow on CPU for large `n_iterations`; recommended: start with 50–100 epochs.
```

---
