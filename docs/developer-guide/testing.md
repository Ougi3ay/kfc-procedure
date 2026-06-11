# Testing and Quality Assurance


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## Observed test result

```text
============================= 116 passed in 7.43s ==============================
```

The test suite was executed with:

```bash
cd kfc-procedure
PYTHONPATH=src pytest -q tests/cobra
```

## Test coverage observed from file structure

| Test file | Area covered |
|---|---|
| `test_p01_splitters.py` | holdout and overlap splitters |
| `test_p02_estimators.py` | mean estimator and sklearn wrapper |
| `test_p03_normalizers.py` | standard and min-max normalizers |
| `test_p04_distances.py` | distance factory and distance matrix behavior |
| `test_p05_kernel_adapters.py` | one- and two-parameter adapters |
| `test_p06_kernels.py` | kernel output behavior |
| `test_p07_losses.py` | loss functions |
| `test_p08_cv.py` | K-fold cross-validation |
| `test_p09_optimizers.py` | gradient, momentum, Adam, and grid search optimizers |
| `test_p10_aggregators.py` | weighted mean and weighted vote aggregators |

## Gaps to address before publication

No dedicated tests were observed for the following high-level package areas:

- `BregmanKMeans`;
- divergence implementations as mathematical units;
- `KStep`;
- `FStep`;
- `CStep`;
- `KFCProcedure`, `KFCRegressor`, and `KFCClassifier`;
- notebooks as reproducible examples.

## Suggested smoke tests

```python
import numpy as np
from kfc_procedure import KFCRegressor, KFCClassifier


def test_kfc_regressor_smoke():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    y = X[:, 0] + rng.normal(scale=0.1, size=80)
    model = KFCRegressor(
        divergences=["euclidean"],
        local_model="linear_regression",
        combiner="mean",
        n_clusters=2,
        random_state=0,
    )
    model.fit(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)


def test_kfc_classifier_smoke():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    y = (X[:, 0] > 0).astype(int)
    model = KFCClassifier(
        divergences=["euclidean"],
        local_model="logistic_regression",
        combiner="majority_vote",
        n_clusters=2,
        random_state=0,
    )
    model.fit(X, y)
    pred = model.predict(X[:5])
    assert len(pred) == 5
```
