# Extending the Package

KFCProcedure is designed to be extended through factories. New components should subclass the proper base class, register themselves, and be imported somewhere during package initialization.

## Add a new divergence

```python
from kfc_procedure.core.clustering.divergences.base import (
    BaseBregmanDivergence,
    BregmanDivergenceFactory,
)

@BregmanDivergenceFactory.register("my_divergence")
class MyDivergence(BaseBregmanDivergence):
    name = "my_divergence"
    family = "custom"

    def in_domain(self, X):
        return True

    def phi(self, X):
        ...

    def grad_phi(self, X):
        ...
```

## Add a new local model

```python
from kfc_procedure.core.ml.base import BaseLocalModel, LocalModelFactory

@LocalModelFactory.register("my_regressor", categories={"regression"})
class MyRegressor(BaseLocalModel):
    def fit(self, X, y):
        return self

    def predict(self, X):
        ...
```

## Add a new combiner

```python
import numpy as np
from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory

@CombinerFactory.register("median", categories={"regression"})
class MedianCombiner(BaseCombiner):
    def fit(self, X, y=None):
        return self

    def combine(self, X):
        return np.median(X, axis=1)
```

## Developer checklist

- Add unit tests for the new component.
- Import the module so the decorator runs.
- Add the component name to documentation.
- Test with `KFCRegressor` or `KFCClassifier`.
