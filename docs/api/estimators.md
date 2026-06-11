# Public Estimators


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## `KFCProcedure`

```python
KFCProcedure(
    divergences,
    local_model,
    combiner,
    divergences_params=None,
    local_model_params=None,
    combiner_params=None,
    task="regression",
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
)
```

Generic clusterwise estimator implementing K-step, F-step, and C-step.

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `divergences` | list | required | Bregman divergence names or divergence objects. |
| `local_model` | str or object | required | Local model name registered in `LocalModelFactory`, or a local model object. |
| `combiner` | str or object | required | Combiner name registered in `CombinerFactory`, or a combiner object. |
| `divergences_params` | dict | `None` | Per-divergence parameter mapping. |
| `local_model_params` | dict | `None` | Parameters passed to the local model constructor. |
| `combiner_params` | dict | `None` | Parameters passed to the combiner constructor. |
| `task` | `('regression', 'classification')` | `"regression"` | Selects task-specific validation and combiner behavior. |
| `n_clusters` | int | `3` | Number of clusters per divergence. |
| `max_iter` | int | `300` | Iteration limit for Bregman K-Means. |
| `tol` | float | `1e-4` | Convergence tolerance. |
| `verbose` | int/bool | `0` | Logger verbosity. |
| `random_state` | int or None | `None` | Reproducibility seed. |

### Methods

```python
fit(X, y)
predict(X)
predict_proba(X)
```

`fit` stores `kstep_`, `fstep_`, and `cstep_`. `predict` returns final aggregated predictions. `predict_proba` is intended for classification only, but the high-level KFC probability path is incomplete in the inspected source.

## `KFCRegressor`

Subclass of `KFCProcedure` that passes `task="regression"`.

## `KFCClassifier`

Subclass of `KFCProcedure` that passes `task="classification"`.

## `GradientCOBRA`

```python
GradientCOBRA(
    estimators=None,
    estimators_params=None,
    distance="euclidean",
    distance_params=None,
    kernel="rbf",
    kernel_params=None,
    aggregator="weighted_mean",
    aggregator_params=None,
    loss="mse",
    loss_params=None,
    optimizer="grid",
    optimizer_params=None,
    opt_method="grid",
    bandwidth_list=None,
    learning_rate=0.1,
    max_iter=300,
    n_cv=5,
    norm_constant=None,
    n_jobs=-1,
    random_state=None,
)
```

Kernel-weighted COBRA-style regressor. It can fit base estimators or consume precomputed prediction features through `fit(..., as_predictions=True)`.

## `MixCOBRARegressor`

COBRA-style regressor combining input-space and prediction-space distances using one- or two-parameter kernel adapters.

## `CombinedClassifier`

```python
CombinedClassifier(
    estimators=None,
    estimators_params=None,
    distance="hamming",
    kernel="rbf",
    aggregator="weighted_vote",
    loss="mse",
    optimizer="grid",
    n_jobs=1,
    bandwidth_list=None,
    max_iter=300,
    n_cv=5,
    random_state=None,
)
```

Classification counterpart that aggregates calibration labels by kernel-weighted voting in prediction space.

## `SuperLearner`

A stacked ensemble implementation present in `kfc_procedure.cobra.superlearner`. It supports base learners, meta learners, cross-validation folds, and loss function configuration. Behavior is documented from source analysis; no dedicated tests for `SuperLearner` were observed.
