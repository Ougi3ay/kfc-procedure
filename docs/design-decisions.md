# Design Decisions


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Composition over inheritance-heavy pipelines

The KFC pipeline is composed of explicit stages: `KStep`, `FStep`, and `CStep`. This is useful because each stage has a distinct responsibility:

- `KStep` creates divergence-specific cluster partitions;
- `FStep` learns cluster-local supervised predictors;
- `CStep` aggregates divergence-level predictions.

This separation makes it easier to test, replace, and explain each part in an academic context.

## Factory-based component resolution

Factories decouple user-facing configuration from implementation classes. A user can write:

```python
KFCRegressor(local_model="ridge", combiner="weighted_mean", divergences=["euclidean"])
```

without importing the concrete classes manually.

## scikit-learn compatibility

Many classes inherit from scikit-learn base classes or follow scikit-learn conventions:

- constructor parameters are stored as attributes;
- fitted attributes use trailing underscores;
- methods such as `fit`, `predict`, `get_params`, and `set_params` are used;
- wrappers adapt scikit-learn estimators into package-specific interfaces.

## Separate COBRA core

The COBRA subsystem is split into small independent pieces rather than a single monolithic estimator. This supports research experiments where a developer may change only the distance, kernel, loss, optimizer, or aggregation rule.

## Known design risk

Factory registries make configuration flexible, but examples can become outdated if aliases are not synchronized. Documentation should therefore show registry-inspection code and use exact names from the current source.
