# Extending Components


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Extension principle

Most extension points are factory-based. A new component should:

1. follow the base class interface;
2. register one or more public names;
3. include task/category metadata when relevant;
4. expose deterministic behavior when `random_state` is provided;
5. include unit tests for constructor, factory creation, and core method behavior.

## Factory interface

Factories expose a shared interface:

```python
Factory.available()
Factory.contains("name")
Factory.create("name", **params)
Factory.register("name", categories={"regression"})
Factory.supports("name", "regression")
```

## Design checklist for new algorithms

| Question | Reason |
|---|---|
| Does the algorithm have a stable public name? | Required for factory registration. |
| Does it support regression, classification, or both? | Needed for category metadata. |
| Does it need input-domain validation? | Required for divergences and distance-like methods. |
| Does it need `random_state`? | Required for reproducible experiments. |
| Does it store learned attributes with trailing underscores? | Maintains scikit-learn style. |
| Does it support `get_params` / `set_params`? | Useful for model selection and cloning. |

## Avoiding API drift

The current README contains short aliases that are not registered in the inspected implementation. Before publishing examples, generate registry names directly:

```python
from kfc_procedure.core.ml.base import LocalModelFactory
from kfc_procedure.core.combiner.base import CombinerFactory

print(LocalModelFactory.available())
print(CombinerFactory.available())
```
