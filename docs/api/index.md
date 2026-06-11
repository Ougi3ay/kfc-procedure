# API Reference


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


This API reference is generated from public classes and functions found under `src/kfc_procedure`. It intentionally excludes private names beginning with `_`. Some scikit-learn metadata-routing helper methods are omitted from method listings because they are inherited/generated compatibility methods rather than package-specific APIs.

## Pages

- [Public Estimators](estimators.md)
- [KFC Core API](kfc-core.md)
- [COBRA Core API](cobra-core.md)
- [Factory Registry Reference](factories.md)
- [Complete Generated API Inventory](generated.md)

## Main imports

```python
from kfc_procedure import KFCProcedure, KFCRegressor, KFCClassifier
from kfc_procedure.cobra import GradientCOBRA, MixCOBRARegressor, CombinedClassifier
```

## Known API caveat

`KFCClassifier.predict_proba()` exists in the high-level class, but the current implementation calls `FStep.predict_proba()`, which is not defined in `FStep`. Treat probability prediction through the high-level KFC pipeline as incomplete until that method is implemented and tested.
