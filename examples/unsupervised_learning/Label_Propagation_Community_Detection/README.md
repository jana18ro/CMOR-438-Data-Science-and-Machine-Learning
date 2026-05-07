# Label Propagation Notebok

# Label_Propagation_Community_Detection/

Demonstrates graph-based community detection using
`rice_ml.unsupervised_learning.LabelPropagation`.

## Notebook

`label_propagation_notebook.ipynb`

## What the Notebook Covers

- **Graph Representation:** Constructing an adjacency matrix and edge list;
  visualising graph structure with NetworkX.
- **Label Propagation Algorithm:** Iterative neighbourhood majority vote —
  step-by-step walkthrough of one iteration.
- **Convergence:** Tracking label stability across iterations; convergence criterion.
- **Community Visualisation:** Colour-coded nodes by detected community on the graph layout.
- **Comparison with Ground Truth:** Comparing detected communities to known cluster memberships
  using modularity score.
- **Stochastic Block Model:** Synthetic graph with planted community structure for
  clean algorithm evaluation.

## Dataset
- Synthetic datasets

## Key imports

```python
from rice_ml.unsupervised_learning import LabelPropagation
import networkx as nx
```
