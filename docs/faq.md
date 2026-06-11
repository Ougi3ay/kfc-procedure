# FAQ


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Why does `pytest` fail with `ModuleNotFoundError`?

The repository uses a `src/` layout. Install the package with `pip install -e .`, or run tests with:

```bash
PYTHONPATH=src python -m pytest
```

## Which local model names should I use?

Use exact names from `LocalModelFactory.available()`. Common valid names include `linear_regression`, `ridge`, `logistic_regression`, `decision_tree_classifier`, `decision_tree_regressor`, `random_forest_classifier`, and `random_forest_regressor`.

## Why does `gkl` or `is` fail?

These divergences require positive input values. If the data include zero, negative values, NaN, or Inf, domain validation raises an error.

## Is `KFCClassifier.predict_proba()` supported?

Not fully in the inspected source. The method exists, but it calls `FStep.predict_proba()`, which is not implemented. Use `KFCClassifier.predict()` or use `CombinedClassifier.predict_proba()` directly.

## Can I use custom scikit-learn estimators?

Yes, components such as COBRA estimators accept estimator objects as well as registered names. For KFC local models, the cleanest route is to wrap or register a `BaseLocalModel`-compatible class.

## How do I make examples reproducible?

Pass `random_state` wherever available and use deterministic data splits. Some underlying scikit-learn estimators also need their own `random_state` in parameter dictionaries.

## Why is fitting slow?

The main causes are multiple divergences, many clusters, expensive local estimators, large pairwise distance matrices, and wide optimizer grids.
