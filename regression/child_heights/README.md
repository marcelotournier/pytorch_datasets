# Child height prediction

This dataset contains the heights of parents, gender of child and the height of the child when adult.

Data is synthetically generated from random normal and Bernoulli distributions, using PyTorch.

To load the data:
```python3
import torch


features = torch.load("height_features.pt")
targets = torch.load("height_targets.pt")
```
