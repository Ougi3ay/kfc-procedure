# Base Estimators

The **base estimator layer** provides the predictive models used by the
F-Step of the KFC Procedure.

The F-Step does not hard-code one regression model or one classifier.
Instead, it resolves local learners through a registry:

```python
LocalModelFactory
```

and trains one model inside every divergence-specific cluster.

\[
\boxed{
\text{cluster data}
\rightarrow
\text{local estimator}
\rightarrow
\text{local prediction}
}
\]

The main implementation lives in:

```text
kfc_procedure/core/ml/
├── base.py
├── sklearn.py
└── __init__.py
```

---

## Main components

The estimator layer contains three important pieces.

<div class="grid cards" markdown>

-   :material-code-braces:{ .lg .middle } **`BaseLocalModel`**

    ---

    Abstract interface used by F-Step local models.

    Requires:

    ```text
    fit()
    predict()
    ```

-   :material-factory:{ .lg .middle } **`LocalModelFactory`**

    ---

    Registry and factory used to resolve estimator names.

    Models are categorized as:

    ```text
    regression
    classification
    ```

-   :material-language-python:{ .lg .middle } **`SklearnLocalModel`**

    ---

    Adapter that wraps scikit-learn estimators so they can be used by the
    F-Step.

</div>

---

# `BaseLocalModel`

All local-model implementations inherit from:

```python
BaseLocalModel
```

which itself inherits:

```python
sklearn.base.BaseEstimator
```

and `ABC`.

The abstract interface is:

```python
class BaseLocalModel(BaseEstimator, ABC):

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def predict_proba(self, X):
        raise NotImplementedError
```

The base class is intentionally task-agnostic.

Task separation is handled by the factory registry.

---

## Required methods

A local model must implement:

```python
fit(X, y)
```

and:

```python
predict(X)
```

The F-Step relies on exactly these methods when training and generating the
prediction matrix consumed by the C-Step.

---

## Optional probability method

The base class also defines:

```python
predict_proba(X)
```

but its default behavior is:

```python
raise NotImplementedError
```

So probability prediction is optional at the local-model level.

The current F-Step itself does not expose a `predict_proba()` method, so
end-to-end KFC classification probabilities are not currently available even
when a local classifier supports probabilities.

---

# `LocalModelFactory`

The registry used by F-Step is:

```python
from kfc_procedure.core.ml import LocalModelFactory
```

It subclasses the package's general-purpose:

```python
BaseFactory
```

and keeps its own independent registry.

The class documentation defines the categories:

```text
regression
classification
multitask
```

although the current built-in registrations primarily use:

```text
regression
classification
```

---

## Why use a registry?

Instead of requiring users to instantiate every estimator manually, KFC can
resolve a model from a string:

```python
local_model="ridge"
```

or:

```python
local_model="logistic_regression"
```

F-Step then asks the factory to create the matching implementation.

Conceptually:

```text
"ridge"
   │
   ▼
LocalModelFactory
   │
   ▼
SklearnLocalModel(Ridge)
   │
   ▼
fit on one cluster
```

---

# Automatic scikit-learn registration

When this module is imported:

```python
from kfc_procedure.core.ml import ...
```

the package automatically calls:

```python
register_all_sklearn_models()
```

The registration function uses:

```python
sklearn.utils.all_estimators()
```

to discover available scikit-learn estimators.

It keeps estimator classes that:

```text
have fit()
and inherit ClassifierMixin or RegressorMixin
```

Those estimators are then registered automatically with
`LocalModelFactory`.

---

## Registration categories

A discovered estimator is registered as:

```text
classification
```

if it subclasses:

```python
ClassifierMixin
```

and otherwise as:

```text
regression
```

if it subclasses:

```python
RegressorMixin
```

This category information is later used by F-Step to prevent task mismatch.

---

# Estimator names

Scikit-learn class names are converted to lower-case snake case.

The conversion is performed by:

```python
clean_sklearn_name()
```

Examples:

| Scikit-learn class | Registry name |
| --- | --- |
| `LinearRegression` | `linear_regression` |
| `LogisticRegression` | `logistic_regression` |
| `RandomForestRegressor` | `random_forest_regressor` |
| `RandomForestClassifier` | `random_forest_classifier` |
| `KNeighborsRegressor` | `k_neighbors_regressor` |
| `KNeighborsClassifier` | `k_neighbors_classifier` |
| `DecisionTreeClassifier` | `decision_tree_classifier` |
| `RidgeCV` | `ridge_cv` |
| `LassoCV` | `lasso_cv` |
| `SVC` | `svc` |
| `SVR` | `svr` |

All-uppercase names are simply lowercased.

For example:

```text
SVC -> svc
SVR -> svr
```

---

# The exact available model list depends on scikit-learn

Because KFC discovers estimators dynamically with:

```python
all_estimators()
```

the complete set of model names depends on the installed scikit-learn
version.

For that reason, application code and documentation should not rely on a
permanently fixed exhaustive list.

Instead, inspect the registry in the installed environment.

---

## List all registered local models

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

print(
    LocalModelFactory.available()
)
```

This returns a sorted list of all registered names and aliases.

---

## List regression models

```python
print(
    LocalModelFactory.available_by_category(
        "regression"
    )
)
```

---

## List classification models

```python
print(
    LocalModelFactory.available_by_category(
        "classification"
    )
)
```

This is the recommended way to discover the models actually available in the
installed package.

---

# Common regression estimators

The dynamically registered estimator set commonly includes names such as:

```text
linear_regression
ridge
ridge_cv
lasso
lasso_cv
elastic_net
elastic_net_cv
decision_tree_regressor
random_forest_regressor
extra_trees_regressor
gradient_boosting_regressor
hist_gradient_boosting_regressor
k_neighbors_regressor
svr
dummy_regressor
```

!!! note

    This is an illustrative subset.

    The exact available names should be obtained from:

    ```python
    LocalModelFactory.available_by_category(
        "regression"
    )
    ```

---

# Common classification estimators

Typical dynamically registered names include:

```text
logistic_regression
decision_tree_classifier
random_forest_classifier
extra_trees_classifier
gradient_boosting_classifier
hist_gradient_boosting_classifier
k_neighbors_classifier
svc
linear_svc
gaussian_nb
dummy_classifier
```

Again, use the registry to inspect the actual installed set.

---

# Built-in `MeanRegressor`

In addition to dynamically registered scikit-learn models, the source
explicitly registers:

```text
mean_regressor
dummy_mean
```

as aliases for:

```python
MeanRegressor
```

The registration is:

```python
@LocalModelFactory.register(
    "mean_regressor",
    "dummy_mean",
    categories={"regression"},
)
```

The model wraps:

```python
DummyRegressor(
    strategy="mean"
)
```

---

## Mean-regressor behavior

Its implementation is:

```python
class MeanRegressor(BaseLocalModel):

    def __init__(self):
        self.estimator = DummyRegressor(
            strategy="mean"
        )

    def fit(self, X, y):
        self.estimator.fit(
            X,
            y,
        )
        return self

    def predict(self, X):
        return np.asarray(
            self.estimator.predict(X),
            dtype=float,
        )
```

So each local cluster predicts the mean target learned from that cluster.

---

## Current `MeanRegressor` integration issue

The current F-Step automatically injects:

```python
random_state=self.random_state
```

when a local model is resolved from a string.

However:

```python
MeanRegressor.__init__()
```

currently accepts no parameters.

Therefore a string-based configuration such as:

```python
local_model="mean_regressor"
```

can receive an unsupported `random_state` keyword when created by F-Step.

!!! warning "Current source behavior"

    In this source version, `mean_regressor` / `dummy_mean` may raise a
    constructor `TypeError` when selected through F-Step.

    This is an implementation issue, not a conceptual limitation of a
    cluster-wise mean predictor.

---

# `SklearnLocalModel`

Scikit-learn estimators are not registered directly.

Instead, each dynamically registered name points to a small builder that
returns:

```python
SklearnLocalModel(
    model_cls,
    **kwargs,
)
```

This adapter gives scikit-learn models the `BaseLocalModel` interface expected
by F-Step.

---

## Constructor

The adapter receives:

```python
model_cls
```

and arbitrary keyword arguments:

```python
SklearnLocalModel(
    model_cls,
    **kwargs,
)
```

It then inspects:

```python
model_cls.__init__
```

using:

```python
inspect.signature()
```

and retains only keyword arguments that are explicitly present in the
constructor signature.

---

## Parameter filtering

The implementation performs:

```python
valid_kwargs = {
    key: value
    for key, value in kwargs.items()
    if key in signature.parameters
}
```

and then:

```python
self.model = model_cls(
    **valid_kwargs
)
```

This has an important user-facing consequence.

Unknown parameters are silently removed instead of raising an error at this
adapter layer.

---

## Example

Suppose:

```python
local_model_params={
    "alpha": 2.0,
    "not_a_real_parameter": 99,
}
```

and the selected estimator only accepts `alpha`.

The adapter keeps:

```python
alpha=2.0
```

and discards:

```python
not_a_real_parameter
```

!!! warning "Check parameter spelling"

    A typo in a scikit-learn constructor parameter can therefore be silently
    ignored.

    Inspect the fitted local estimator's parameters if configuration accuracy
    matters.

---

# Inspect estimator parameters

After KFC has fitted local models, retrieve one:

```python
local_model = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]
```

Then:

```python
print(
    local_model.get_params()
)
```

For `SklearnLocalModel`, `get_params()` delegates directly to:

```python
self.model.get_params()
```

---

## Change parameters with `set_params()`

The adapter also delegates:

```python
set_params(...)
```

to the underlying scikit-learn estimator.

```python
local_model.set_params(
    alpha=2.0
)
```

This matches the standard scikit-learn parameter API.

---

# Access the wrapped estimator

The underlying estimator is stored as:

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

You can then inspect estimator-specific fitted state:

```python
print(
    wrapped.model.coef_
)
```

when the selected estimator defines that attribute.

---

# Probability capability

During initialization, `SklearnLocalModel` records whether the wrapped
estimator exposes:

```python
predict_proba
```

with:

```python
self._has_proba = hasattr(
    self.model,
    "predict_proba",
)
```

Then:

```python
local_model.predict_proba(X)
```

delegates to the wrapped estimator when available.

Otherwise it raises:

```text
<EstimatorName> does not support predict_proba
```

---

## Example

A model such as logistic regression normally supports:

```python
predict_proba()
```

while some classifiers do not.

The adapter handles this capability check dynamically.

!!! important

    This local capability does not currently make
    `KFCClassifier.predict_proba()` functional, because the current `FStep`
    itself has no `predict_proba()` method.

---

# `random_state` handling

The F-Step resolves string models with:

```python
params = dict(
    self.local_model_params
)

if "random_state" not in params:
    params["random_state"] = self.random_state
```

Therefore the top-level KFC seed is attempted for every string-based local
model.

For scikit-learn-backed estimators, the adapter filters it out when the
constructor does not accept `random_state`.

---

## Example

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 300,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

F-Step creates each random forest using the equivalent of:

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
)
```

---

## Local seed overrides top-level seed

If you explicitly pass:

```python
local_model_params={
    "random_state": 123,
}
```

the F-Step does not replace it.

So:

```text
local_model_params["random_state"]
```

takes precedence over:

```text
KFC random_state
```

---

# Task validation

Before creating a string-based local model, F-Step checks:

```python
LocalModelFactory.supports(
    name,
    self.task,
)
```

This prevents incompatible estimator categories.

---

## Regression mismatch example

This is invalid:

```python
KFCRegressor(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="gradientcobra",
)
```

because:

```text
logistic_regression
```

is registered as:

```text
classification
```

---

## Classification mismatch example

This is invalid:

```python
KFCClassifier(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="combined_classifier",
)
```

because:

```text
ridge
```

is registered as:

```text
regression
```

---

# Check whether a model supports a task

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

print(
    LocalModelFactory.supports(
        "ridge",
        "regression",
    )
)
```

Expected:

```text
True
```

For:

```python
print(
    LocalModelFactory.supports(
        "ridge",
        "classification",
    )
)
```

expected:

```text
False
```

---

# Check whether a model exists

```python
print(
    LocalModelFactory.contains(
        "random_forest_regressor"
    )
)
```

This returns a Boolean rather than raising for invalid input names.

---

# Inspect registry metadata

The underlying factory provides:

```python
LocalModelFactory.info(name)
```

Example:

```python
info = LocalModelFactory.info(
    "ridge"
)

print(info)
```

The result contains:

```text
name
class
module
categories
metadata
```

For dynamically registered scikit-learn estimators, the registered target is
the builder function created by the registration loop.

---

# Registry normalization

`BaseFactory` normalizes registration names by:

```text
stripping surrounding whitespace
converting to lowercase
```

So lookups are case-insensitive with respect to registry normalization.

Conceptually:

```python
LocalModelFactory.contains(
    "RIDGE"
)
```

resolves the normalized key:

```text
ridge
```

---

# Registry aliases

The factory can register multiple names for the same target.

For example:

```text
mean_regressor
dummy_mean
```

refer to the same built-in `MeanRegressor` class.

The general factory also provides:

```python
find_by_class(...)
```

to retrieve aliases registered for a target class.

---

# Registry categories

Inspect all categories:

```python
print(
    LocalModelFactory.available_categories()
)
```

For the current local-model registry, the important categories are:

```text
regression
classification
```

---

# F-Step creates one estimator per cluster

When `local_model` is a string, F-Step calls:

```python
LocalModelFactory.create(...)
```

inside the loop over divergence/cluster pairs.

Therefore each cluster receives a new local-model instance.

This is an important property.

For:

```python
divergences=[
    "euclidean",
    "gkl",
]

n_clusters=3
```

the F-Step can create:

```text
euclidean / cluster 0 -> new estimator
euclidean / cluster 1 -> new estimator
euclidean / cluster 2 -> new estimator

gkl / cluster 0       -> new estimator
gkl / cluster 1       -> new estimator
gkl / cluster 2       -> new estimator
```

The estimators have the same class and constructor configuration but learn
independently from different subsets.

---

# One model type across all clusters

The current F-Step accepts one global:

```python
local_model
```

and one global:

```python
local_model_params
```

Therefore you cannot currently configure:

```text
Ridge for Euclidean clusters
RandomForest for GKL clusters
SVR for Logistic clusters
```

within one F-Step instance.

All divergence/cluster combinations use the same estimator type and parameter
configuration.

---

# Pre-instantiated estimator objects

F-Step also accepts a non-string local model object:

```python
local_model=my_model
```

The relevant source logic is:

```python
if not isinstance(
    self.local_model,
    str,
):
    return self.local_model
```

This means the exact same object instance is returned for every cluster.

---

## Important reuse behavior

Suppose:

```python
my_model = MyEstimator()

model = KFCRegressor(
    divergences=["euclidean"],
    local_model=my_model,
    combiner="gradientcobra",
)
```

During F-Step fitting, the same object can be fitted repeatedly:

```text
cluster 0 -> my_model.fit(...)
cluster 1 -> same my_model.fit(...)
cluster 2 -> same my_model.fit(...)
```

The current F-Step does not clone the object.

!!! warning "Prefer registered string estimators"

    For independent cluster models, use a registered string name.

    Pre-instantiated local-model objects are reused rather than cloned in the
    current implementation.

---

# Custom local estimator

A custom estimator can subclass:

```python
BaseLocalModel
```

and implement:

```python
fit()
predict()
```

For example:

```python
import numpy as np

from kfc_procedure.core.ml import (
    BaseLocalModel,
)


class MedianRegressor(
    BaseLocalModel
):

    def fit(
        self,
        X,
        y,
    ):
        self.value_ = float(
            np.median(y)
        )
        return self

    def predict(
        self,
        X,
    ):
        return np.full(
            len(X),
            self.value_,
            dtype=float,
        )
```

---

# Register a custom estimator

Register it with:

```python
LocalModelFactory
```

using a task category.

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)


@LocalModelFactory.register(
    "median_regressor",
    categories={"regression"},
)
class MedianRegressor(
    BaseLocalModel
):
    ...
```

Then use:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="median_regressor",
    combiner="gradientcobra",
)
```

---

## Custom constructor compatibility

Remember that F-Step adds:

```python
random_state=...
```

to every string-based local-model constructor unless it is already present in
`local_model_params`.

Therefore a directly registered custom estimator should normally support:

```python
def __init__(
    self,
    random_state=None,
    ...
):
    ...
```

or accept arbitrary keyword arguments.

Otherwise factory creation can fail with:

```text
unexpected keyword argument 'random_state'
```

The scikit-learn adapter avoids this problem by filtering unsupported kwargs,
but directly registered custom classes do not receive that filtering
automatically.

---

# Register multiple aliases

The factory supports multiple registration names:

```python
@LocalModelFactory.register(
    "median_regressor",
    "median",
    categories={"regression"},
)
class MedianRegressor(
    BaseLocalModel
):
    ...
```

Then both names create the same class.

---

# Duplicate-name protection

`BaseFactory.register()` normalizes names and checks for conflicts before
registration.

Attempting to register a name that already exists raises:

```python
KeyError
```

This protects the estimator registry from silent replacement.

---

# Registration names are normalized

The factory:

1. strips surrounding whitespace;
2. converts names to lower case;
3. rejects empty names;
4. rejects duplicates in the same registration call.

So:

```text
"  MyModel  "
```

becomes:

```text
mymodel
```

---

# Creating an estimator directly from the factory

Although normal KFC users let F-Step do this, you can create a model directly.

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

model = LocalModelFactory.create(
    "ridge",
    alpha=2.0,
)
```

For a dynamically registered scikit-learn estimator, this returns a
`SklearnLocalModel` wrapper.

Then:

```python
model.fit(
    X_train,
    y_train,
)

prediction = model.predict(
    X_test,
)
```

---

# Inspect the created wrapper

```python
local_model = LocalModelFactory.create(
    "ridge",
    alpha=2.0,
)

print(
    type(local_model).__name__
)
```

The wrapper type is:

```text
SklearnLocalModel
```

The underlying estimator can be accessed through:

```python
print(
    local_model.model
)
```

---

# Example: regression estimator

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

ridge = LocalModelFactory.create(
    "ridge",
    alpha=1.5,
)

ridge.fit(
    X_train,
    y_train,
)

y_pred = ridge.predict(
    X_test,
)
```

---

# Example: classification estimator

```python
logit = LocalModelFactory.create(
    "logistic_regression",
    C=1.0,
    max_iter=1000,
)

logit.fit(
    X_train,
    y_train,
)

y_pred = logit.predict(
    X_test,
)
```

If supported:

```python
proba = logit.predict_proba(
    X_test,
)
```

---

# Estimator parameters in KFC

Use:

```python
local_model_params
```

to configure a string-based estimator.

Regression:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="random_forest_regressor",
    local_model_params={
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 3,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

Classification:

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

For more detail, see:

[Estimator Parameters](estimator-parameters.md)

---

# Choosing a regression estimator

The source does not prescribe one estimator as universally best.

The F-Step architecture is designed so that a local model can be selected
according to the application.

A practical code-oriented interpretation is:

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Linear models**

    ---

    Examples:

    ```text
    linear_regression
    ridge
    lasso
    ```

    Useful when relationships inside clusters are approximately simple.

-   :material-tree:{ .lg .middle } **Tree models**

    ---

    Examples:

    ```text
    decision_tree_regressor
    random_forest_regressor
    ```

    Useful when local relationships remain nonlinear.

-   :material-map-marker-distance:{ .lg .middle } **Neighborhood models**

    ---

    Example:

    ```text
    k_neighbors_regressor
    ```

-   :material-vector-polyline:{ .lg .middle } **Kernel / margin models**

    ---

    Example:

    ```text
    svr
    ```

</div>

These are usage categories, not package-imposed rankings.

---

# Choosing a classification estimator

Likewise, classification can use any registered classifier compatible with the
cluster-level target distribution.

Common choices include:

```text
logistic_regression
decision_tree_classifier
random_forest_classifier
k_neighbors_classifier
svc
```

Remember that some classifiers cannot fit when a local cluster contains only
one class.

---

# Small-cluster behavior

Every F-Step model is trained only on its assigned cluster.

Therefore the effective sample size can be much smaller than the full training
sample.

If:

```text
n_clusters
```

is large, local estimators may receive very few observations.

This can cause:

- unstable regression models;
- one-class classification clusters;
- neighbor-based models with too few samples;
- estimator-specific validation errors.

The F-Step catches `ValueError` raised during local-model fitting and rethrows
it with divergence and cluster information.

---

# F-Step error message

The source wraps fitting errors as:

```text
[FSTEP ERROR] divergence='...', cluster=... failed.
Reason: ...
Hint: cluster contains invalid label distribution.
```

The hint is always included, even if the original `ValueError` is not related
to classification.

Inspect the underlying `Reason:` text for the actual estimator failure.

---

# Scikit-learn estimator version sensitivity

Because registration is dynamic:

```python
all_estimators()
```

the exact local-model registry can change when the installed scikit-learn
version changes.

This affects:

- available estimator names;
- estimator constructor parameters;
- default values;
- supported capabilities such as `predict_proba()`.

For stable production configurations, pin the scikit-learn dependency version.

---

# Quick registry inspection

A useful diagnostic script is:

```python
from kfc_procedure.core.ml import (
    LocalModelFactory,
)

print("Categories:")
print(
    LocalModelFactory.available_categories()
)

print("\nRegression:")
print(
    LocalModelFactory.available_by_category(
        "regression"
    )
)

print("\nClassification:")
print(
    LocalModelFactory.available_by_category(
        "classification"
    )
)
```

---

# Inspect one registration

```python
name = "random_forest_regressor"

print(
    LocalModelFactory.info(name)
)
```

and:

```python
print(
    LocalModelFactory.supports(
        name,
        "regression",
    )
)
```

---

# Quick reference

| Component | Role |
| --- | --- |
| `BaseLocalModel` | abstract local-estimator interface |
| `LocalModelFactory` | name/category registry |
| `SklearnLocalModel` | scikit-learn adapter |
| `register_all_sklearn_models()` | dynamic estimator registration |
| `clean_sklearn_name()` | converts sklearn class names to registry names |
| `MeanRegressor` | explicitly registered local mean baseline |

---

# Important current-source behaviors

| Behavior | Current implementation |
| --- | --- |
| sklearn estimator discovery | dynamic via `all_estimators()` |
| task classification | `ClassifierMixin` / `RegressorMixin` |
| estimator naming | snake-case class names |
| unsupported sklearn kwargs | silently filtered |
| top-level `random_state` | injected by F-Step |
| unsupported `random_state` on sklearn class | filtered by adapter |
| pre-instantiated local model | reused, not cloned |
| local model type per cluster | one shared configuration for all clusters |
| local `predict_proba()` | supported by adapter when underlying model supports it |
| F-Step `predict_proba()` | not currently implemented |
| `mean_regressor` alias | `mean_regressor`, `dummy_mean` |
| `mean_regressor` through F-Step | currently affected by `random_state` constructor mismatch |

---

# Mental model

!!! quote ""

    **The estimator registry answers “what model should each cluster use?”;
    the F-Step answers “which fitted instance should handle this observation?”**

\[
\boxed{
\text{registered estimator name}
\rightarrow
\text{new local model per cluster}
\rightarrow
\text{cluster-specific fit}
\rightarrow
\text{F-Step prediction}
}
\]

