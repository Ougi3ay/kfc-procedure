# Inspect Fitted KFC Models

After `KFCRegressor` or `KFCClassifier` has been fitted, the three trained
stages are available directly from the estimator:

```python
model.kstep_
model.fstep_
model.cstep_
```

These attributes expose the fitted clustering models, local predictive models,
and final combiner.

This page shows how to inspect each stage and how to reconstruct the
intermediate predictions that move through the KFC pipeline.

---

## Fit a model first

For regression:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="gradientcobra",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)
```

For classification:

```python
from kfc_procedure import KFCClassifier

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="combined_classifier",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)
```

After `fit()`, KFC checks and uses the fitted attributes:

```text
kstep_
fstep_
cstep_
```

during prediction.

---

## Pipeline inspection map

```mermaid
flowchart LR
    X["New X"]

    K["model.kstep_"]
    C["cluster assignments"]

    F["model.fstep_"]
    P["prediction matrix"]

    S["model.cstep_.strategy_"]
    Y["final prediction"]

    X --> K --> C --> F --> P --> S --> Y
```

A useful debugging pattern is therefore:

```python
clusters = model.kstep_.predict(X_test)

P = model.fstep_.predict(
    X_test,
    clusters,
)

y_pred = model.cstep_.predict(P)
```

This should reproduce:

```python
model.predict(X_test)
```

for the same fitted model.

---

# Inspect the K-Step

The fitted clustering stage is:

```python
kstep = model.kstep_
```

Its main fitted attributes are:

```text
models_
clusters_
```

---

## `kstep_.models_`

`models_` is a dictionary containing one fitted `BregmanKMeans` instance per
divergence.

```python
print(
    model.kstep_.models_.keys()
)
```

For example:

```text
dict_keys([
    "euclidean",
    "gkl",
    "logistic",
    "is",
])
```

when those four divergence strings were configured.

Inspect the model types:

```python
for divergence, clustering_model in model.kstep_.models_.items():
    print(
        divergence,
        type(clustering_model).__name__,
    )
```

Typical output:

```text
euclidean BregmanKMeans
gkl       BregmanKMeans
logistic  BregmanKMeans
is        BregmanKMeans
```

---

## Inspect cluster centers

Each fitted `BregmanKMeans` stores:

```python
cluster_centers_
```

Example:

```python
km = model.kstep_.models_["euclidean"]

print(
    km.cluster_centers_
)
```

Its shape is:

```text
(n_clusters, n_features)
```

You can inspect the dimensions:

```python
print(
    km.cluster_centers_.shape
)
```

---

## Inspect training cluster labels

The underlying clustering model stores:

```python
labels_
```

Example:

```python
labels = model.kstep_.models_[
    "euclidean"
].labels_

print(
    labels[:20]
)
```

These labels correspond to the internal K-Step training subset used during
`fit()`.

---

## `kstep_.clusters_`

K-Step also stores all training assignments in:

```python
model.kstep_.clusters_
```

This is a dictionary keyed by divergence.

```python
for divergence, labels in model.kstep_.clusters_.items():
    print(
        divergence,
        labels.shape,
    )
```

You can inspect cluster counts:

```python
import numpy as np

for divergence, labels in model.kstep_.clusters_.items():
    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(
        divergence,
        dict(
            zip(
                unique,
                counts,
            )
        ),
    )
```

This is especially useful when diagnosing very small clusters.

---

## Inspect cluster distortion

Each `BregmanKMeans` stores:

```python
inertia_
```

In the current implementation, this is the best run's **average Bregman
distortion**.

```python
for divergence, km in model.kstep_.models_.items():
    print(
        divergence,
        km.inertia_,
    )
```

!!! note

    Do not assume that `inertia_` values from different divergence families
    are directly comparable as though they were measured on the same metric
    scale.

---

## Inspect iteration count

The selected clustering run stores:

```python
n_iter_
```

Example:

```python
for divergence, km in model.kstep_.models_.items():
    print(
        divergence,
        "iterations=",
        km.n_iter_,
    )
```

---

## Inspect all main BregmanKMeans state

```python
for divergence, km in model.kstep_.models_.items():
    print(
        "\nDivergence:",
        divergence,
    )

    print(
        "centers:",
        km.cluster_centers_.shape,
    )

    print(
        "labels:",
        km.labels_.shape,
    )

    print(
        "distortion:",
        km.inertia_,
    )

    print(
        "iterations:",
        km.n_iter_,
    )
```

---

# Predict and inspect new cluster assignments

For new observations:

```python
clusters = model.kstep_.predict(
    X_test
)
```

The result is a dictionary:

```python
{
    "euclidean": ...,
    "gkl": ...,
}
```

Inspect it:

```python
for divergence, labels in clusters.items():
    print(
        divergence,
        labels[:10],
    )
```

Every value has shape:

```text
(n_samples,)
```

---

## Inspect distances to centroids

The underlying `BregmanKMeans` provides:

```python
transform(X)
```

For example:

```python
km = model.kstep_.models_["euclidean"]

D = km.transform(
    X_test
)

print(
    D.shape
)
```

With three clusters:

```text
(n_test, 3)
```

Each row contains the divergence from one observation to each fitted centroid.

The predicted cluster is the minimum-distance centroid.

```python
manual_labels = np.argmin(
    D,
    axis=1,
)

automatic_labels = km.predict(
    X_test
)

print(
    np.array_equal(
        manual_labels,
        automatic_labels,
    )
)
```

---

# Inspect the F-Step

The fitted local-model stage is:

```python
fstep = model.fstep_
```

Its main fitted attribute is:

```python
models_
```

---

## Structure of `fstep_.models_`

The current source stores a nested dictionary:

```text
models_
└── divergence
    └── model key such as "m0"
        ├── divergence
        ├── cluster
        └── model
```

Example:

```python
{
    "euclidean": {
        "m0": {
            "divergence": "euclidean",
            "cluster": 0,
            "model": ...
        },
        "m1": {
            "divergence": "euclidean",
            "cluster": 1,
            "model": ...
        }
    }
}
```

Inspect the top-level keys:

```python
print(
    model.fstep_.models_.keys()
)
```

---

## List all fitted local models

```python
for divergence, models in model.fstep_.models_.items():
    print(
        "\nDivergence:",
        divergence,
    )

    for model_name, metadata in models.items():
        print(
            model_name,
            "cluster=",
            metadata["cluster"],
            "type=",
            type(
                metadata["model"]
            ).__name__,
        )
```

---

## Count local models

```python
for divergence, models in model.fstep_.models_.items():
    print(
        divergence,
        len(models),
    )
```

Total count:

```python
total = sum(
    len(models)
    for models in model.fstep_.models_.values()
)

print(
    "Total local models:",
    total,
)
```

With \(M\) divergences and \(K\) populated clusters per divergence, this will
usually be approximately:

\[
M\times K.
\]

---

## Get one local model

For example:

```python
metadata = model.fstep_.models_[
    "euclidean"
]["m0"]

local_model = metadata["model"]

print(
    local_model
)
```

Inspect its cluster ID:

```python
print(
    metadata["cluster"]
)
```

and divergence:

```python
print(
    metadata["divergence"]
)
```

---

## Inspect local-model parameters

For scikit-learn-backed local models, the wrapper exposes:

```python
get_params()
```

Example:

```python
local_model = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

print(
    local_model.get_params()
)
```

This is useful for confirming that options in:

```python
local_model_params
```

were actually forwarded to the underlying estimator.

---

## Inspect the wrapped scikit-learn model

When the local estimator is represented by the package's scikit-learn
adapter, it stores the underlying estimator on:

```python
local_model.model
```

Example:

```python
wrapped = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

print(
    type(
        wrapped.model
    ).__name__
)
```

Then normal scikit-learn fitted attributes may be available.

For example, for linear or logistic regression:

```python
print(
    wrapped.model.coef_
)
```

when the underlying estimator defines `coef_`.

!!! note

    The exact fitted attributes depend on the selected local estimator.

    KFC itself does not normalize them into one common interface.

---

# Reconstruct the F-Step prediction matrix

The C-Step does not see the original feature matrix.

It sees the prediction matrix created by F-Step.

To inspect it:

```python
clusters = model.kstep_.predict(
    X_test
)

P_test = model.fstep_.predict(
    X_test,
    clusters,
)

print(
    P_test.shape
)

print(
    P_test[:5]
)
```

If four divergences were configured:

```text
P_test.shape == (n_test, 4)
```

---

## Regression example

The matrix might look like:

```text
[[12.7, 13.1, 12.4, 12.9],
 [18.4, 17.9, 18.8, 18.2],
 [ 5.6,  5.9,  5.5,  5.7]]
```

Each column corresponds to one divergence-specific local-model family.

---

## Classification example

```text
[[0, 0, 1, 0],
 [1, 1, 1, 1],
 [2, 1, 2, 2]]
```

The C-Step combines each row into one final class.

---

## Verify the full prediction path

You can manually reconstruct `model.predict()`.

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

manual_prediction = model.cstep_.predict(
    P
)

normal_prediction = model.predict(
    X_test
)
```

For numeric regression output:

```python
import numpy as np

print(
    np.allclose(
        manual_prediction,
        normal_prediction,
    )
)
```

For classification:

```python
print(
    np.array_equal(
        manual_prediction,
        normal_prediction,
    )
)
```

Expected:

```text
True
```

---

# Check the F-Step matrix for missing values

The F-Step initializes each divergence prediction vector with:

```python
np.nan
```

and fills entries for matching fitted cluster models.

For numeric regression predictions:

```python
print(
    np.isnan(P_test).any()
)
```

Normally:

```text
False
```

If `True`, inspect:

```python
clusters
model.fstep_.models_
```

and confirm that every prediction-time cluster ID has a corresponding stored
local model.

---

# Inspect the C-Step

The fitted aggregation stage is:

```python
cstep = model.cstep_
```

Its fitted strategy is:

```python
model.cstep_.strategy_
```

Inspect its type:

```python
strategy = model.cstep_.strategy_

print(
    type(strategy).__name__
)
```

The exact fitted attributes depend on the combiner.

---

# Inspect `mean`

`MeanCombiner` is stateless.

There are no learned coefficients to inspect.

```python
print(
    type(
        model.cstep_.strategy_
    ).__name__
)
```

The combiner simply computes the row-wise mean at prediction time.

---

# Inspect `weighted_mean`

`WeightedMeanCombiner` stores a scikit-learn linear regression instance on:

```python
strategy.model
```

Example:

```python
strategy = model.cstep_.strategy_

print(
    strategy.model.coef_
)

print(
    strategy.model.intercept_
)
```

The coefficients correspond to the F-Step prediction columns.

If the F-Step columns correspond to:

```text
euclidean
gkl
logistic
is
```

then the coefficient vector follows that same column order.

!!! note

    The current `weighted_mean` implementation uses ordinary linear
    regression.

    The coefficients are not constrained to be non-negative and are not
    constrained to sum to one.

---

# Inspect `stacking_regressor`

The fitted clone of the regression meta-model is:

```python
strategy.meta_model_
```

Example:

```python
strategy = model.cstep_.strategy_

print(
    strategy.meta_model_
)
```

If it is linear regression:

```python
print(
    strategy.meta_model_.coef_
)
```

---

# Inspect `stacking_classifier`

Similarly:

```python
strategy = model.cstep_.strategy_

print(
    strategy.meta_model_
)
```

For the default logistic regression meta-model:

```python
print(
    strategy.meta_model_.classes_
)

print(
    strategy.meta_model_.coef_
)
```

---

# Inspect `gradientcobra`

The KFC regression wrapper stores the fitted `GradientCOBRA` object on:

```python
strategy.cobra
```

Example:

```python
strategy = model.cstep_.strategy_

cobra = strategy.cobra

print(
    cobra
)
```

Useful fitted attributes can include:

```python
cobra.bandwidth_
cobra.distance_matrix_
cobra.cv_folds_
cobra.optimization_outputs_
cobra.global_mean_
```

depending on the fitted source version and configuration.

Example:

```python
print(
    cobra.bandwidth_
)

print(
    cobra.optimization_outputs_
)
```

Optimization history:

```python
print(
    cobra.optimization_outputs_[
        "history"
    ]
)
```

---

# Inspect `mixcobra`

The KFC regression wrapper stores:

```python
strategy.cobra
```

which is a fitted `MixCOBRARegressor`.

Example:

```python
cobra = model.cstep_.strategy_.cobra

print(
    cobra.optimization_outputs_
)
```

Depending on mode, useful attributes can include:

```text
distance_matrix_x_
distance_matrix_y_
optimization_outputs_
global_mean_
```

and the normalization constants used internally.

Because KFC passes the F-Step matrix using:

```python
as_predictions=True
```

interpret these attributes in the context of the wrapper's precomputed
prediction-space use.

---

# Inspect `combined_classifier`

The classification COBRA wrapper also stores:

```python
strategy.cobra
```

Example:

```python
cobra = model.cstep_.strategy_.cobra

print(
    cobra.bandwidth_
)

print(
    cobra.classes_
)

print(
    cobra.global_majority_class_
)

print(
    cobra.optimization_outputs_
)
```

Its aggregation prediction-space representation is stored as:

```python
cobra.pred_l_
```

and the pairwise aggregation distance matrix is:

```python
cobra.distance_matrix_
```

when fitted.

---

# Inspect majority vote

`MajorityVoteCombiner` is stateless.

There are no learned parameters.

You can inspect its input directly:

```python
P = model.fstep_.predict(
    X_test,
    model.kstep_.predict(
        X_test
    ),
)

print(
    P[:10]
)
```

and compare it with:

```python
print(
    model.cstep_.predict(P)[:10]
)
```

This is often the most useful way to understand majority-vote decisions.

---

# Inspect a single prediction end to end

For one sample:

```python
x = X_test[:1]
```

### 1. Cluster assignments

```python
clusters = model.kstep_.predict(
    x
)

print(
    clusters
)
```

Example:

```python
{
    "euclidean": array([1]),
    "gkl": array([0]),
}
```

### 2. F-Step predictions

```python
P = model.fstep_.predict(
    x,
    clusters,
)

print(
    P
)
```

Example regression output:

```text
[[12.4, 13.0]]
```

or classification:

```text
[[0, 1]]
```

### 3. Final C-Step output

```python
final = model.cstep_.predict(
    P
)

print(
    final
)
```

This three-stage inspection is the simplest way to explain one KFC prediction.

---

# Inspect one divergence path

Suppose you want to trace only the Euclidean path.

```python
x = X_test[:1]

euclidean_km = model.kstep_.models_[
    "euclidean"
]

cluster_id = euclidean_km.predict(
    x
)[0]

print(
    "cluster:",
    cluster_id,
)
```

Find the corresponding local model:

```python
local_models = model.fstep_.models_[
    "euclidean"
]

selected = None

for metadata in local_models.values():
    if metadata["cluster"] == cluster_id:
        selected = metadata["model"]
        break
```

Then:

```python
local_prediction = selected.predict(
    x
)

print(
    local_prediction
)
```

That value should match the Euclidean column of the F-Step prediction matrix.

---

# Map F-Step columns to divergence names

The F-Step prediction columns follow the insertion order of:

```python
fstep.models_
```

You can recover the names with:

```python
divergence_names = list(
    model.fstep_.models_.keys()
)

print(
    divergence_names
)
```

Then:

```python
P = model.fstep_.predict(
    X_test,
    model.kstep_.predict(
        X_test
    ),
)
```

and inspect one sample:

```python
row = P[0]

for name, value in zip(
    divergence_names,
    row,
):
    print(
        name,
        value,
    )
```

This is especially useful when interpreting C-Step coefficients.

---

# Build a small inspection table

For regression:

```python
import pandas as pd

clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

columns = list(
    model.fstep_.models_.keys()
)

table = pd.DataFrame(
    P,
    columns=columns,
)

table["final_prediction"] = model.cstep_.predict(
    P
)

print(
    table.head()
)
```

This produces a table conceptually like:

```text
   euclidean    gkl   logistic   is   final_prediction
0      12.4     12.8      12.1  12.5      12.45
1      20.7     21.0      20.1  20.8      20.65
```

---

# Add cluster assignments to the table

```python
for divergence, labels in clusters.items():
    table[
        f"{divergence}_cluster"
    ] = labels
```

Then:

```python
print(
    table.head()
)
```

Now the table shows both routing and prediction behavior.

---

# Classification inspection table

For classification:

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

columns = list(
    model.fstep_.models_.keys()
)

table = pd.DataFrame(
    P,
    columns=[
        f"{name}_prediction"
        for name in columns
    ],
)

table["final_prediction"] = model.cstep_.predict(
    P
)

print(
    table.head()
)
```

---

# What KFC does not retain at the top level

The current `KFCProcedure.fit()` creates local variables:

```text
X_k
X_l
y_k
y_l
clusters_k
clusters_l
P_l
```

but does **not** store them on the top-level estimator.

So after fitting, you cannot directly access:

```python
model.X_k_
model.X_l_
model.y_k_
model.y_l_
model.P_l_
```

because those attributes are not created by the current source.

!!! important

    If you need the exact internal split for auditing or diagnostics, reproduce
    it externally using the same `train_test_split` arguments, or modify the
    implementation to retain those arrays.

---

# Reproduce the internal split

For regression:

```python
from sklearn.model_selection import train_test_split

X_k, X_l, y_k, y_l = train_test_split(
    X_train,
    y_train,
    test_size=0.5,
    random_state=model.random_state,
    stratify=None,
)
```

For classification:

```python
X_k, X_l, y_k, y_l = train_test_split(
    X_train,
    y_train,
    test_size=0.5,
    random_state=model.random_state,
    stratify=y_train,
)
```

With the same input ordering, seed, and scikit-learn environment, this follows
the same splitting call used by `KFCProcedure.fit()`.

---

# Reconstruct the C-Step training matrix

Once the internal split is reproduced:

```python
clusters_l = model.kstep_.predict(
    X_l
)

P_l = model.fstep_.predict(
    X_l,
    clusters_l,
)
```

Then:

```python
print(
    P_l.shape
)
```

This is the kind of matrix used to train:

```python
model.cstep_.strategy_
```

during the original fit.

!!! note

    `P_l` itself is not stored by `KFCProcedure`.

---

# Check whether the estimator is fitted

KFC's top-level `predict()` calls:

```python
check_is_fitted(
    self,
    [
        "kstep_",
        "fstep_",
        "cstep_",
    ]
)
```

You can do the same:

```python
from sklearn.utils.validation import check_is_fitted

check_is_fitted(
    model,
    [
        "kstep_",
        "fstep_",
        "cstep_",
    ],
)
```

If no exception is raised, all three top-level fitted stages exist.

---

# Inspect constructor configuration

Because `KFCProcedure` inherits from scikit-learn `BaseEstimator`, normal
estimator parameter inspection is available:

```python
print(
    model.get_params()
)
```

This reports constructor configuration such as:

```text
divergences
local_model
combiner
divergences_params
local_model_params
combiner_params
n_clusters
max_iter
tol
verbose
random_state
```

This is different from fitted state.

Use:

```text
get_params()
```

for configuration and the trailing-underscore attributes for learned state.

---

# Inspect KFC logging configuration

The estimator stores:

```python
model.verbose
model.logger
```

The logger is created during construction from the supplied `verbose` value.

This is configuration state, not fitted state, but it can help when
reproducing debugging behavior.

---

# Regression inspection example

```python
import numpy as np
import pandas as pd

from kfc_procedure import KFCRegressor


model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="gradientcobra",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


# K-Step
for name, km in model.kstep_.models_.items():
    print(
        name,
        km.cluster_centers_.shape,
        km.inertia_,
        km.n_iter_,
    )


# F-Step
for name, local_models in model.fstep_.models_.items():
    print(
        name,
        len(local_models),
    )


# Intermediate predictions
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)


# C-Step
strategy = model.cstep_.strategy_

print(
    type(strategy).__name__
)


# Final prediction
y_pred = model.cstep_.predict(
    P
)


# Inspection table
table = pd.DataFrame(
    P,
    columns=list(
        model.fstep_.models_.keys()
    ),
)

table["prediction"] = y_pred

print(
    table.head()
)
```

---

# Classification inspection example

```python
import pandas as pd

from kfc_procedure import KFCClassifier


model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="combined_classifier",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

final = model.cstep_.predict(
    P
)


table = pd.DataFrame(
    P,
    columns=[
        f"{name}_prediction"
        for name
        in model.fstep_.models_.keys()
    ],
)

table["final_prediction"] = final

print(
    table.head()
)
```

---

# Probability inspection limitation

The top-level classifier source defines:

```python
KFCClassifier.predict_proba(...)
```

through inheritance from `KFCProcedure`.

That implementation attempts to call:

```python
self.fstep_.predict_proba(
    X,
    clusters,
)
```

but the current `FStep` implementation does not define `predict_proba()`.

Therefore end-to-end probability inspection is not currently available through:

```python
model.predict_proba(X)
```

in this source version.

The `combined_classifier` C-Step wrapper itself does expose a probability
method on its wrapped classifier, but the top-level KFC probability pipeline
fails earlier at F-Step.

---

# Useful inspection checklist

When a fitted KFC model behaves unexpectedly, inspect the pipeline in this
order:

```text
1. K-Step models_
       ↓
2. cluster counts
       ↓
3. cluster centers and inertia_
       ↓
4. F-Step models_
       ↓
5. local model types and parameters
       ↓
6. prediction matrix P
       ↓
7. C-Step strategy_
       ↓
8. combiner-specific fitted attributes
       ↓
9. final prediction
```

This follows the same direction as the actual prediction pipeline.

---

# Quick reference

| Object | Useful fitted state |
| --- | --- |
| `model.kstep_` | `models_`, `clusters_` |
| `BregmanKMeans` | `labels_`, `cluster_centers_`, `inertia_`, `n_iter_` |
| `model.fstep_` | `models_` |
| local-model metadata | `divergence`, `cluster`, `model` |
| `model.cstep_` | `strategy_` |
| weighted mean | `strategy_.model.coef_` |
| stacking | `strategy_.meta_model_` |
| GradientCOBRA wrapper | `strategy_.cobra` |
| MixCOBRA wrapper | `strategy_.cobra` |
| CombinedClassifier wrapper | `strategy_.cobra` |

---

# Mental model

!!! quote ""

    **Inspect KFC in the same order that data flows through it: clusters,
    local predictions, then consensus.**

\[
\boxed{
X
\rightarrow
\underbrace{\text{K-Step}}_{\text{where?}}
\rightarrow
\underbrace{\text{F-Step}}_{\text{what local prediction?}}
\rightarrow
\underbrace{\text{C-Step}}_{\text{how combined?}}
\rightarrow
\widehat y
}
\]
