# Child height prediction

This dataset contains the heights of parents, gender of child and the height of the child when adult.

Data is synthetically generated from random normal and Bernoulli distributions, using PyTorch, using [./make_dataset.py](this script).

To load the data:
```python3
import torch


features = torch.load("height_features.pt")
targets = torch.load("height_targets.pt")
```

### Metadata
The `features` data contains 100,000 observations. Each "row" has the following schema:
```
[mother_height, father_height, is_female]
```
- `mother_height` (float) - Mother's height as adult, in centimeters
- `father_height` (float) - Mother's height as adult, in centimeters
- `is_female` (float) - Indicator if the child is genetically female (1.0 = yes, 0.0 = no)

In `targets`, we also have 100,000 observations. Each "row" contains the expected height of the child when adult, in centimeters.
