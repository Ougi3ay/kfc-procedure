# KFC Core API


!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.


## `BregmanKMeans`

```python
BregmanKMeans(
    n_clusters=8,
    *,
    divergence,
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=None,
    verbose=False,
)
```

Lloyd-style clustering with pluggable Bregman divergences.

### Attributes

| Attribute | Description |
|---|---|
| `labels_` | Cluster assignment for each training sample. |
| `cluster_centers_` | Final centroids. |
| `inertia_` | Distortion value for the selected initialization. |
| `n_iter_` | Number of iterations for the selected initialization. |

### Methods

```python
fit(X, y=None)
predict(X)
fit_predict(X, y=None)
transform(X)
```

Input data are validated against the selected divergence domain. Log-based divergences require positive or bounded input domains.

## `KStep`

```python
KStep(
    divergences,
    divergences_params={},
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    verbose=False,
    random_state=None,
)
```

Fits one `BregmanKMeans` model per divergence and stores:

- `models_`: fitted clustering models keyed by divergence name;
- `clusters_`: training labels keyed by divergence name.

## `FStep`

```python
FStep(
    local_model,
    local_model_params=None,
    task="regression",
    random_state=None,
)
```

Fits one local model for each `(divergence, cluster)` pair. `predict` returns a matrix of shape `(n_samples, n_divergences)`. Each column corresponds to predictions produced under one divergence-specific partition.

## `CStep`

```python
CStep(
    combiner,
    combiner_params=None,
    task="regression",
    random_state=None,
)
```

Fits and applies an aggregation strategy to the F-step prediction matrix.

## Divergence classes

| Class | Factory name | Domain |
|---|---|---|
| `SquaredEuclidean` | `euclidean` | Real-valued data |
| `GKLDivergence` | `gkl` | Positive data |
| `ItakuraSaito` | `is` | Positive data |
| `LogisticLoss` | `logistic` | Values in `(0, 1)` |

## KFC combiner classes

| Class | Task | Factory name |
|---|---|---|
| `MeanCombiner` | regression | `mean` |
| `WeightedMeanCombiner` | regression | `weighted_mean` |
| `StackingRegressorCombiner` | regression | `stacking_regressor` |
| `GradientCOBRACombiner` | regression | `gradientcobra` |
| `MixCOBRACombiner` | regression | `mixcobra` |
| `MajorityVoteCombiner` | classification | `majority_vote` |
| `StackingClassifierCombiner` | classification | `stacking_classifier` |
| `CobraClassifierCombiner` | classification | `combined_classifier` |
