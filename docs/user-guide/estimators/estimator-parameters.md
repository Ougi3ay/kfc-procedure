# Estimator Parameters

This guide explains how parameters are passed from `KFCRegressor` and
`KFCClassifier` to the local estimators fitted by the F-Step.

The main user-facing argument is:

```python
local_model_params
```

For example:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    local_model_params={
        "alpha": 2.0,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

The F-Step uses the same local-model configuration for every
divergence/cluster pair.

---

## Parameter flow

The parameter path is:

```text
KFCRegressor / KFCClassifier
            │
            ▼
local_model_params
            │
            ▼
FStep
            │
            ▼
LocalModelFactory
            │
            ▼
SklearnLocalModel
            │
            ▼
scikit-learn estimator constructor
```

For a string-based local model, the current F-Step resolves parameters using:

```python
params = dict(
    self.local_model_params
)

if "random_state" not in params:
    params["random_state"] = self.random_state

return LocalModelFactory.create(
    name,
    **params,
)
```

So the top-level KFC `random_state` is added automatically unless you provide a
model-specific value.

---

# Basic usage

Regression:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    local_model_params={
        "alpha": 1.5,
        "fit_intercept": True,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

Classification:

```python
from kfc_procedure import KFCClassifier

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    local_model_params={
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    combiner="combined_classifier",
    random_state=42,
)
```

---

# The same parameters are used in every cluster

The current F-Step has one global:

```python
local_model
```

and one global:

```python
local_model_params
```

Therefore:

```python
local_model="ridge"

local_model_params={
    "alpha": 2.0,
}
```

means:

```text
Euclidean / cluster 0 -> Ridge(alpha=2.0)
Euclidean / cluster 1 -> Ridge(alpha=2.0)
Euclidean / cluster 2 -> Ridge(alpha=2.0)

GKL / cluster 0       -> Ridge(alpha=2.0)
GKL / cluster 1       -> Ridge(alpha=2.0)
GKL / cluster 2       -> Ridge(alpha=2.0)
```

The current F-Step does not provide separate parameter dictionaries per
divergence or per cluster.

---

# How scikit-learn parameters are filtered

Dynamically registered scikit-learn estimators are wrapped by:

```python
SklearnLocalModel
```

The adapter inspects the estimator constructor with:

```python
inspect.signature(
    model_cls.__init__
)
```

and keeps only keyword arguments whose names appear in that signature:

```python
valid_kwargs = {
    key: value
    for key, value in kwargs.items()
    if key in sig.parameters
}
```

Then it constructs:

```python
self.model = model_cls(
    **valid_kwargs
)
```

This behavior applies to estimators discovered through
`sklearn.utils.all_estimators()`.

---

## Unsupported parameters are silently dropped

Consider:

```python
local_model="ridge"

local_model_params={
    "alpha": 2.0,
    "not_a_parameter": 123,
}
```

If `Ridge.__init__()` does not contain:

```text
not_a_parameter
```

then the adapter simply removes it.

The estimator is effectively constructed with:

```python
Ridge(
    alpha=2.0,
)
```

!!! warning "Typos can be silent"

    Because unsupported scikit-learn constructor keywords are filtered out,
    a misspelled parameter may not raise an error.

    For example:

    ```python
    local_model_params={
        "n_estimator": 300,
    }
    ```

    can be silently ignored when the estimator actually expects:

    ```python
    n_estimators
    ```

    Inspect the fitted estimator when parameter correctness matters.

---

# Inspect the effective parameters

After fitting, retrieve one local estimator:

```python
local_model = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]
```

For a `SklearnLocalModel`, call:

```python
print(
    local_model.get_params()
)
```

The adapter delegates this call to the underlying scikit-learn estimator:

```python
return self.model.get_params(
    deep=deep
)
```

This is the simplest way to confirm which parameters were actually accepted.

---

## Example

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 300,
        "max_depth": 8,
        "not_real": 999,
    },
    combiner="gradientcobra",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

local_model = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

params = local_model.get_params()

print(
    params["n_estimators"]
)

print(
    params["max_depth"]
)

print(
    "not_real" in params
)
```

Expected behavior:

```text
300
8
False
```

---

# Access the underlying estimator

For dynamically registered scikit-learn models, the wrapper stores the real
estimator on:

```python
local_model.model
```

Example:

```python
wrapped = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

print(
    wrapped.model
)
```

This is useful when you want estimator-specific fitted attributes such as:

```python
wrapped.model.coef_
wrapped.model.feature_importances_
wrapped.model.classes_
```

depending on the selected estimator.

---

# `random_state`

The F-Step automatically adds:

```python
random_state=self.random_state
```

when the local parameter dictionary does not already contain that key.

So:

```python
model = KFCRegressor(
    ...,
    random_state=42,
)
```

attempts to configure every string-based local model with:

```python
random_state=42
```

---

## If the estimator supports `random_state`

For example:

```python
local_model="random_forest_regressor"
```

with:

```python
random_state=42
```

is effectively constructed with:

```python
RandomForestRegressor(
    random_state=42,
    ...
)
```

assuming you did not provide another value in `local_model_params`.

---

## If the estimator does not support `random_state`

For dynamically wrapped scikit-learn estimators, the adapter filters the
unsupported keyword out.

For example, if the selected estimator constructor has no `random_state`
parameter, this injected value is removed before construction.

This means the top-level KFC seed can be forwarded broadly without breaking
most dynamically registered scikit-learn models.

---

## Override the top-level seed

A value supplied directly through:

```python
local_model_params
```

takes precedence.

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 300,
        "random_state": 123,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

Here the local forests receive:

```python
random_state=123
```

rather than:

```python
42
```

because F-Step only injects its value when the key is absent.

---

# Regression examples

## Ridge

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    local_model_params={
        "alpha": 2.0,
        "fit_intercept": True,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

---

## Lasso

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="lasso",
    local_model_params={
        "alpha": 0.01,
        "max_iter": 5000,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

---

## Random forest

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    combiner="gradientcobra",
    random_state=42,
)
```

---

## k-nearest neighbors

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="k_neighbors_regressor",
    local_model_params={
        "n_neighbors": 5,
        "weights": "distance",
    },
    combiner="gradientcobra",
)
```

`KNeighborsRegressor` does not require `random_state`; the injected value is
filtered out by `SklearnLocalModel`.

---

## SVR

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="svr",
    local_model_params={
        "C": 10.0,
        "epsilon": 0.1,
        "kernel": "rbf",
    },
    combiner="gradientcobra",
)
```

---

# Classification examples

## Logistic regression

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    local_model_params={
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    combiner="combined_classifier",
    random_state=42,
)
```

---

## Decision tree

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="decision_tree_classifier",
    local_model_params={
        "max_depth": 6,
        "min_samples_leaf": 3,
    },
    combiner="combined_classifier",
    random_state=42,
)
```

---

## Random forest classifier

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="random_forest_classifier",
    local_model_params={
        "n_estimators": 300,
        "max_depth": 8,
        "class_weight": "balanced",
    },
    combiner="combined_classifier",
    random_state=42,
)
```

---

## K-nearest neighbors classifier

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="k_neighbors_classifier",
    local_model_params={
        "n_neighbors": 7,
        "weights": "distance",
    },
    combiner="combined_classifier",
)
```

---

## SVC

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="svc",
    local_model_params={
        "C": 1.0,
        "kernel": "rbf",
    },
    combiner="combined_classifier",
    random_state=42,
)
```

If you need probability support directly on the wrapped SVC, configure the
underlying estimator accordingly:

```python
local_model_params={
    "C": 1.0,
    "kernel": "rbf",
    "probability": True,
}
```

The current top-level KFC probability pipeline still has a separate F-Step
limitation described in the classification guide.

---

# Parameters are constructor parameters

`local_model_params` is forwarded when the local model is **created**.

It is not applied after fitting.

Conceptually:

```python
LocalModelFactory.create(
    name,
    **local_model_params,
)
```

happens before:

```python
model.fit(
    X_cluster,
    y_cluster,
)
```

So values in `local_model_params` should match the selected estimator's
constructor arguments.

---

# Fit-time arguments are not supported separately

The current F-Step always calls:

```python
model.fit(
    Xc,
    yc,
)
```

with only two arguments.

There is no current top-level F-Step parameter for passing extra fit-time
arguments such as:

```text
sample_weight
callbacks
eval_set
```

to each local estimator.

!!! note "Current API scope"

    `local_model_params` configures constructors only.

    It is not a general `fit()` keyword dictionary.

---

# One parameter dictionary for all local fits

Suppose:

```python
local_model="random_forest_regressor"

local_model_params={
    "n_estimators": 300,
    "max_depth": 8,
}
```

Every cluster receives the same configuration.

The current source does not support:

```text
cluster 0 -> max_depth=4
cluster 1 -> max_depth=8
cluster 2 -> max_depth=None
```

nor:

```text
Euclidean -> Ridge(alpha=1)
GKL       -> Ridge(alpha=5)
```

inside one F-Step.

If you require those behaviors, the F-Step would need to be extended.

---

# Task categories still apply

Changing parameters does not change the estimator's registered task category.

For example:

```python
local_model="ridge"
```

remains a regression estimator regardless of its parameter values.

F-Step validates:

```python
LocalModelFactory.supports(
    name,
    task,
)
```

before model construction.

So parameter customization cannot make a regression model valid for
classification or vice versa.

---

# Discover the selected estimator

Because KFC auto-registers scikit-learn models, the exact available set can
depend on the installed scikit-learn version.

Check whether a model exists:

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

print(
    LocalModelFactory.contains(
        "hist_gradient_boosting_regressor"
    )
)
```

---

## List available regression estimators

```python
print(
    LocalModelFactory.available_by_category(
        "regression"
    )
)
```

---

## List available classifiers

```python
print(
    LocalModelFactory.available_by_category(
        "classification"
    )
)
```

This should be done before assuming a particular estimator name is available in
every environment.

---

# Inspect registry metadata

```python
info = LocalModelFactory.info(
    "ridge"
)

print(
    info
)
```

The factory metadata identifies:

```text
name
class
module
categories
metadata
```

For dynamically registered scikit-learn estimators, the registered target is
the builder function created during automatic registration.

---

# Direct factory creation

You can test a parameter configuration without running the full KFC pipeline.

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

local = LocalModelFactory.create(
    "random_forest_regressor",
    n_estimators=300,
    max_depth=8,
    random_state=42,
)

print(
    local.get_params()
)
```

This is useful for validating which parameters survive the adapter filtering.

---

## Check a suspected typo

```python
local = LocalModelFactory.create(
    "random_forest_regressor",
    n_estimator=300,
    random_state=42,
)

params = local.get_params()

print(
    "n_estimator" in params
)

print(
    params["n_estimators"]
)
```

The typo is not retained.

The underlying estimator uses its own default for `n_estimators`.

---

# Parameter changes after creation

`SklearnLocalModel` exposes:

```python
set_params(...)
```

and delegates directly to the wrapped estimator.

Example:

```python
local = LocalModelFactory.create(
    "ridge",
    alpha=1.0,
)

local.set_params(
    alpha=3.0,
)
```

Then:

```python
print(
    local.get_params()["alpha"]
)
```

returns:

```text
3.0
```

---

## Changing fitted KFC local models

You can technically call:

```python
set_params()
```

on a local model extracted from:

```python
model.fstep_.models_
```

but changing estimator parameters does not automatically refit that local
model or the rest of the KFC pipeline.

For example:

```python
local_model.set_params(
    alpha=3.0
)
```

does not retrain it.

To obtain a coherent KFC model, update the top-level configuration and run:

```python
model.fit(...)
```

again.

---

# Inspect all fitted local-model parameters

You can audit every cluster model after training.

```python
for divergence, models in model.fstep_.models_.items():
    print(
        "\n",
        divergence,
    )

    for model_name, metadata in models.items():
        local = metadata["model"]

        print(
            model_name,
            "cluster=",
            metadata["cluster"],
        )

        if hasattr(
            local,
            "get_params",
        ):
            print(
                local.get_params()
            )
```

This is useful when verifying reproducibility.

---

# Compare requested and effective parameters

A simple audit pattern is:

```python
requested = {
    "n_estimators": 300,
    "max_depth": 8,
    "random_state": 42,
}

local = LocalModelFactory.create(
    "random_forest_regressor",
    **requested,
)

effective = local.get_params()

for key, value in requested.items():
    print(
        key,
        "requested=",
        value,
        "effective=",
        effective.get(
            key,
            "<filtered>",
        ),
    )
```

---

# Parameter compatibility depends on the installed estimator version

Because the adapter examines the actual constructor signature:

```python
inspect.signature(
    model_cls.__init__
)
```

a parameter is retained only if the installed estimator version declares it.

This means behavior can change when scikit-learn changes estimator signatures.

For reproducible production use, pin the scikit-learn version together with
the KFC package version.

---

# Directly registered custom models behave differently

The filtering described above belongs specifically to:

```python
SklearnLocalModel
```

A custom model registered directly with:

```python
LocalModelFactory.register(...)
```

receives factory keyword arguments directly.

For example:

```python
@LocalModelFactory.register(
    "my_model",
    categories={"regression"},
)
class MyModel(BaseLocalModel):

    def __init__(
        self,
        scale=1.0,
        random_state=None,
    ):
        ...
```

F-Step can create:

```python
MyModel(
    scale=2.0,
    random_state=42,
)
```

---

## Unsupported custom-model kwargs are not filtered automatically

If your directly registered custom constructor is:

```python
def __init__(
    self,
    scale=1.0,
):
    ...
```

but F-Step creates it with:

```python
random_state=42
```

then normal Python constructor behavior applies and a `TypeError` can be
raised.

The factory itself forwards arguments directly:

```python
target_cls(
    *args,
    **kwargs,
)
```

It does not inspect or remove unsupported keywords.

---

# Recommended custom-model constructor pattern

Because F-Step injects `random_state`, a direct custom local estimator should
usually accept it:

```python
class MyRegressor(
    BaseLocalModel
):

    def __init__(
        self,
        scale=1.0,
        random_state=None,
    ):
        self.scale = scale
        self.random_state = random_state
```

Then:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="my_regressor",
    local_model_params={
        "scale": 2.0,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

works with the F-Step parameter path.

---

# Current `mean_regressor` issue

The built-in `MeanRegressor` constructor is currently:

```python
def __init__(
    self,
) -> None:
    ...
```

It does not accept:

```python
random_state
```

But F-Step injects that key for every string model.

Because `MeanRegressor` is directly registered rather than wrapped in
`SklearnLocalModel`, unsupported arguments are not filtered.

Therefore:

```python
local_model="mean_regressor"
```

or:

```python
local_model="dummy_mean"
```

can fail during factory construction in the current source version.

!!! warning "Implementation-specific issue"

    The problem is parameter forwarding, not the behavior of
    `DummyRegressor(strategy="mean")` itself.

---

# Nested estimator parameters

Some scikit-learn estimators can expose nested parameters through:

```python
get_params(
    deep=True
)
```

and standard scikit-learn `set_params()` can use names such as:

```text
component__parameter
```

However, the KFC `SklearnLocalModel` constructor filtering step compares
`local_model_params` only with the selected estimator's direct
`__init__()` signature.

Therefore arbitrary nested `__` parameters are not a general constructor-time
mechanism provided by the current KFC adapter.

Use the selected estimator's documented direct constructor arguments, or
provide a custom registered wrapper if you need a more specialized composite
estimator configuration.

---

# Parameter tuning

The F-Step does not perform local-estimator hyperparameter search.

For example:

```python
local_model="ridge"
local_model_params={
    "alpha": 1.0,
}
```

uses exactly that constructor configuration for every cluster.

KFC does not automatically compare:

```text
alpha=0.1
alpha=1.0
alpha=10.0
```

for the F-Step.

---

## Use estimators that tune themselves

Because scikit-learn estimators are registered dynamically, estimators with
built-in tuning APIs can be selected when available.

For example:

```python
local_model="ridge_cv"
```

or:

```python
local_model="lasso_cv"
```

can perform their own internal estimator-specific tuning when fitted.

Their accepted constructor parameters still follow the installed
scikit-learn class signature.

---

## External tuning

You can also treat KFC constructor settings as ordinary estimator
hyperparameters and evaluate different complete KFC configurations outside the
F-Step.

The source itself does not provide a dedicated search utility for
`local_model_params`.

---

# Common mistakes

## Wrong parameter name

```python
local_model_params={
    "n_estimator": 300,
}
```

instead of:

```python
local_model_params={
    "n_estimators": 300,
}
```

may be silently ignored by `SklearnLocalModel`.

---

## Passing fit-time options as constructor parameters

```python
local_model_params={
    "sample_weight": weights,
}
```

does not make F-Step call:

```python
fit(
    X,
    y,
    sample_weight=weights,
)
```

The current F-Step always calls:

```python
fit(
    Xc,
    yc,
)
```

---

## Assuming parameters differ by cluster

One `local_model_params` dictionary applies to all local model instances.

---

## Assuming top-level `random_state` always controls randomness

It is forwarded only when:

```python
"random_state"
```

is absent from `local_model_params`.

Also, estimators without a `random_state` constructor parameter simply discard
the injected value through the scikit-learn adapter.

---

## Assuming accepted parameters are stable across scikit-learn versions

The adapter uses the constructor signature present in the installed
environment.

Pin dependencies when exact configuration behavior matters.

---

# Debugging parameter configuration

## 1. Inspect requested parameters

```python
print(
    model.local_model_params
)
```

---

## 2. Inspect one fitted local model

```python
local = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

print(
    local
)
```

---

## 3. Inspect effective parameters

```python
print(
    local.get_params()
)
```

---

## 4. Inspect the wrapped estimator

```python
print(
    local.model
)
```

for a `SklearnLocalModel`.

---

## 5. Compare all cluster configurations

```python
for divergence, models in model.fstep_.models_.items():
    for name, metadata in models.items():
        local = metadata["model"]

        print(
            divergence,
            name,
            local.get_params()
            if hasattr(
                local,
                "get_params",
            )
            else "<no get_params>",
        )
```

---

# Quick reference

| Question | Current behavior |
| --- | --- |
| Where do local parameters go? | estimator constructor |
| Same parameters for every cluster? | Yes |
| Same parameters for every divergence? | Yes |
| Is `random_state` added automatically? | Yes, if absent |
| Are unsupported sklearn kwargs rejected? | No, silently filtered |
| Are unsupported custom-model kwargs filtered? | No |
| Are fit-time kwargs supported? | No |
| Are local parameters tuned automatically? | No |
| Can fitted sklearn parameters be inspected? | Yes, via `get_params()` |
| Can the wrapped estimator be inspected? | Yes, via `.model` |
| Are object local models cloned? | No |

---

# Mental model

!!! quote ""

    **`local_model_params` describes how every cluster-local estimator is
    constructed; F-Step then creates one independently fitted instance per
    cluster when the model is selected by registry name.**

\[
\boxed{
\text{local_model_params}
\rightarrow
\text{constructor filtering}
\rightarrow
\text{new estimator}
\rightarrow
\text{cluster-specific fit}
}
\]

