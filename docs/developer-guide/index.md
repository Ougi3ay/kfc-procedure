# Developer Guide


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Repository layout

```text
kfc-procedure/
├── pyproject.toml
├── requirements.txt
├── README.md
├── notebooks/
├── src/
│   └── kfc_procedure/
│       ├── kfc.py
│       ├── core/
│       ├── cobra/
│       └── utils/
└── tests/
    └── cobra/
```

## Coding standards observed

The source follows a scikit-learn-like estimator style:

- constructors store parameters without performing heavy work;
- learned attributes use trailing underscores, for example `models_`, `clusters_`, `strategy_`, `bandwidth_`;
- methods return `self` after fitting;
- validation uses `np.asarray`, `check_array`, and `check_is_fitted` in several components;
- component discovery uses case-insensitive factory registries.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cobra]"
PYTHONPATH=src python -m pytest tests/cobra
```

## Adding a new Bregman divergence

1. Create a new class under `src/kfc_procedure/core/clustering/divergences/`.
2. Subclass `BaseBregmanDivergence`.
3. Implement `phi`, `grad_phi`, `distance`, `centroid`, and `in_domain` as required by the base class pattern.
4. Register the class with `BregmanDivergenceFactory.register(...)`.
5. Add tests for domain validation, pairwise distance shape, and use through `BregmanKMeans`.

Skeleton:

```python
from kfc_procedure.core.clustering.divergences.base import BaseBregmanDivergence, BregmanDivergenceFactory

@BregmanDivergenceFactory.register("my_divergence")
class MyDivergence(BaseBregmanDivergence):
    name = "my_divergence"

    def phi(self, X):
        ...

    def grad_phi(self, X):
        ...

    def in_domain(self, X):
        ...
```

## Adding a new local model

Use `LocalModelFactory.register` if the model is not automatically registered from scikit-learn.

```python
from kfc_procedure.core.ml.base import BaseLocalModel, LocalModelFactory

@LocalModelFactory.register("my_model", categories={"regression"})
class MyLocalModel(BaseLocalModel):
    def fit(self, X, y):
        ...
        return self

    def predict(self, X):
        ...
```

## Adding a new combiner

1. Subclass `BaseCombiner`.
2. Implement `fit` and `combine` or `predict` according to the existing combiner style.
3. Register with `CombinerFactory` and include task category metadata.
4. Add tests for shape, deterministic output, and invalid input handling.

## Adding a new COBRA kernel

1. Subclass `BaseKernel` under `cobra/core/kernels/`.
2. Implement the callable behavior used by existing kernels.
3. Register it with `KernelFactory`.
4. Test monotonicity or expected weights for simple distance arrays.

## Testing procedures

Observed test modules cover:

- splitters;
- estimators;
- normalizers;
- distances;
- kernel adapters;
- kernels;
- losses;
- cross-validation;
- optimizers;
- aggregators.

Recommended missing tests:

- Bregman divergence formulas and domains;
- `BregmanKMeans.fit`, `predict`, and empty-cluster behavior;
- `KStep`, `FStep`, and `CStep` integration;
- full `KFCRegressor` and `KFCClassifier` workflows;
- `KFCClassifier.predict_proba()` after implementing `FStep.predict_proba()`;
- notebook smoke tests with small synthetic datasets.

## Build process

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Release checklist

1. Run unit tests.
2. Run minimal examples from this documentation.
3. Update `CHANGELOG.md`.
4. Ensure README aliases match actual factory names.
5. Build the package.
6. Validate package metadata.
7. Tag the release.
