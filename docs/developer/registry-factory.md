# Registry & Factory Pattern

The package uses a shared `BaseFactory` pattern to dynamically register and create components by string name.

## Core factory capabilities

Factories provide:

- `register(*names, categories=...)`
- `create(name, **kwargs)`
- `available()`
- `contains(name)`
- `available_by_category(category)`
- `supports(name, category)`

## Main factories

| Factory | Registered components |
|---|---|
| `BregmanDivergenceFactory` | `euclidean`, `gkl`, `is`, `logistic` |
| `LocalModelFactory` | sklearn local models and `mean_regressor` |
| `CombinerFactory` | regression and classification combiners |
| COBRA factories | estimators, distances, kernels, losses, optimizers, aggregators, splitters |

## Example registration

```python
from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory

@CombinerFactory.register("median", categories={"regression"})
class MedianCombiner(BaseCombiner):
    def fit(self, X, y=None):
        return self

    def combine(self, X):
        import numpy as np
        return np.median(X, axis=1)
```
