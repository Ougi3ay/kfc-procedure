# Limitations & Improvements

This page documents weaknesses inferred from the codebase and recommends future improvements.

---

## 1. Current limitations

### Fixed internal split in KFCProcedure

`KFCProcedure.fit` uses:

```python
train_test_split(..., test_size=0.5)
```

This means half of the data is used for K-step/F-step training and half for C-step calibration. This can be too aggressive for small datasets.

**Improvement:** add a public `split_ratio` or `calibration_size` parameter.

---

### Stateless combiner `random_state` issue

`CStep._build_combiner` injects `random_state` into combiner parameters. This can fail for stateless combiners such as `MeanCombiner` and `MajorityVoteCombiner`, because those classes do not define an `__init__` accepting `random_state` or `**kwargs`.

Observed behavior from a smoke test:

```text
KFCRegressor(..., combiner="mean") -> TypeError
KFCClassifier(..., combiner="majority_vote") -> TypeError
```

**Improvement options:**

1. Add no-op constructors:

```python
class MeanCombiner(BaseCombiner):
    def __init__(self, **kwargs):
        pass
```

```python
class MajorityVoteCombiner(BaseCombiner):
    def __init__(self, **kwargs):
        pass
```

2. Or modify `CStep._build_combiner` so it only injects `random_state` when the target constructor supports it.

---

### `KFCClassifier.predict_proba` calls a missing FStep method

`KFCProcedure.predict_proba` calls:

```python
P = self.fstep_.predict_proba(X, clusters)
```

But the inspected `FStep` implementation defines `fit`, `predict`, and `_resolve`, not `predict_proba`.

**Impact:** `KFCClassifier.predict_proba` can fail even if the local model supports probabilities.

**Improvement:** implement `FStep.predict_proba` that returns a probability tensor or a flattened probability feature matrix compatible with C-step probability combiners.

---

### FStep may output `NaN` for missing cluster models

`FStep.predict` initializes predictions with `NaN`:

```python
pred = np.full(X.shape[0], np.nan)
```

If a sample is assigned to a cluster for which no local model was trained, the prediction remains `NaN`.

**Improvement:** add a fallback model per divergence, such as:

- global model trained on all K/F training samples;
- cluster majority class for classification;
- cluster or global mean for regression.

---

### Non-string local model instances are not cloned

If `local_model` is passed as an already instantiated object, `_resolve` returns the same instance. This can cause the same model object to be refit across clusters.

**Improvement:** use `sklearn.base.clone` when a model instance is passed.

---

### Scikit-learn estimator compatibility of wrappers

`KFCRegressor` and `KFCClassifier` use variadic constructors:

```python
class KFCRegressor(KFCProcedure):
    def __init__(self, *args, **kwargs):
        ...
```

Scikit-learn estimators work best when all constructor parameters are explicit. Variadic arguments may reduce compatibility with tools such as `GridSearchCV`, `get_params`, and model inspection.

**Improvement:** define explicit constructor signatures matching `KFCProcedure`.

---

### No `score` method for KFCProcedure

`BregmanKMeans` implements `score`, but `KFCProcedure` does not expose a task-aware `score` method.

**Improvement:** implement:

- regression score: `R^2` or negative loss;
- classification score: accuracy.

---

### Domain preprocessing is left to the user

Divergences such as `gkl`, `is`, and `logistic` enforce strict domain constraints. The package validates the domain but does not automatically transform data.

**Improvement:** add preprocessing helpers:

- positive shift for GKL/IS;
- min-max scaling to `(0, 1)` for logistic divergence;
- validation utilities that report invalid columns and values.

---

### Bregman centroid update assumes arithmetic mean

The implementation uses arithmetic means for centroid updates. This is correct for many right-sided Bregman clustering settings but should be documented clearly because Bregman centroid behavior depends on divergence orientation and formulation.

**Improvement:** make the divergence orientation explicit and add tests comparing objective decrease across divergences.

---

### COBRA grid search can be expensive

COBRA methods compute calibration pairwise distance matrices and optimize bandwidth or mixing parameters. This can become expensive for large `n_l`.

| Operation | Cost |
|---|---:|
| Calibration distance matrix | `O(n_l^2 q)` |
| Prediction distance matrix | `O(n_test n_l q)` |
| Two-parameter grid search | `O(G_alpha G_beta n_l^2)` |

**Improvement:** add approximate nearest neighbor search, batching, sparse kernels, or FAISS-backed prediction paths for more estimators.

---

### Test coverage focuses mostly on COBRA core

The `tests/cobra/` directory covers splitters, estimators, normalizers, distances, adapters, kernels, losses, CV, optimizers, and aggregators. The KFC core pipeline appears less covered.

**Improvement:** add tests for:

- `BregmanKMeans` convergence and domain validation;
- `KStep` with all divergences;
- `FStep` model training and missing-cluster behavior;
- `CStep` combiner compatibility;
- end-to-end `KFCRegressor` and `KFCClassifier` workflows.

---

## 2. Future improvements

### API improvements

- Add explicit constructor signatures for `KFCRegressor` and `KFCClassifier`.
- Add `split_ratio` to `KFCProcedure`.
- Add `score` methods.
- Add `get_feature_names_out` for prediction matrices.
- Add a clean `Pipeline` integration example with scikit-learn preprocessing.

### Modeling improvements

- Support soft cluster assignments.
- Add a global fallback model.
- Add automatic divergence selection.
- Add cross-validation for `n_clusters`.
- Add model pruning for weak divergence views.
- Add probability-aware F-step outputs for classification.

### Performance improvements

- Batch F-step predictions.
- Parallelize local model training across divergence/cluster pairs.
- Support sparse feature matrices where possible.
- Use approximate nearest neighbors for COBRA prediction.
- Cache distance matrices more systematically.

### Documentation improvements

- Add examples for each divergence.
- Add examples for custom divergence registration.
- Add examples for custom local models and custom combiners.
- Add a troubleshooting page based on actual error messages.
- Add API pages using `mkdocstrings`.

---

## 3. Recommended bug-fix patch ideas

### Fix stateless combiners

```python
@CombinerFactory.register("mean", categories={"regression"})
class MeanCombiner(BaseCombiner):
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def combine(self, X):
        return np.mean(np.asarray(X), axis=1)
```

```python
@CombinerFactory.register("majority_vote", categories={"classification"})
class MajorityVoteCombiner(BaseCombiner):
    def __init__(self, **kwargs):
        pass
```

### Add FStep.predict_proba

A simple design is:

```text
For each divergence:
    For each cluster model:
        if model supports predict_proba:
            fill probability rows for samples assigned to cluster
        else:
            raise AttributeError
Return probability features suitable for the chosen combiner
```

The exact output shape must be coordinated with `CStep.predict_proba`.

### Add clone for local model instances

```python
from sklearn.base import clone

if not isinstance(self.local_model, str):
    return clone(self.local_model)
```

---

## 4. Production readiness checklist

Before using this library in production, validate:

- all selected divergences match the feature domain;
- clusters have enough samples for the selected local model;
- no `NaN` values appear in the F-step prediction matrix;
- chosen combiner works with the current task;
- results are reproducible with `random_state`;
- memory usage is acceptable for COBRA distance matrices;
- end-to-end tests pass for the target workflow.
