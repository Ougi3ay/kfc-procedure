# Custom Estimators

You can extend the KFC F-Step with your own local regression or classification
models.

The extension point is:

```python
BaseLocalModel
```

together with:

```python
LocalModelFactory
```

A custom local estimator participates in the same F-Step workflow as built-in
and dynamically registered scikit-learn estimators:

\[
\boxed{
\text{cluster data}
\rightarrow
\text{custom local model}
\rightarrow
\text{F-Step prediction matrix}
}
\]

The relevant source modules are:

```text
kfc_procedure/core/ml/base.py
kfc_procedure/core/ml/sklearn.py
kfc_procedure/core/steps/fstep.py
kfc_procedure/factory/base.py
```

---

## Two ways to use a custom estimator

There are two supported patterns:

1. **register a custom model by name** and let F-Step create one instance per
   cluster;
2. **pass a pre-instantiated model object** directly.

The first approach is usually safer because F-Step creates a new registered
model for every divergence/cluster pair.

---

# Option 1 — Register a custom model

A custom estimator should subclass:

```python
BaseLocalModel
```

and implement:

```python
fit()
predict()
```

For classification, you may also implement:

```python
predict_proba()
```

although the current F-Step itself does not yet expose a probability-prediction
method.

---

## Minimal custom regressor

```python
import numpy as np

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "median_regressor",
    categories={"regression"},
)
class MedianRegressor(BaseLocalModel):

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

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

After registration:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="median_regressor",
    combiner="gradientcobra",
    n_clusters=3,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

y_pred = model.predict(
    X_test
)
```

---

# Why `random_state` should usually be accepted

When F-Step resolves a model by string, it runs:

```python
params = dict(
    self.local_model_params
)

if "random_state" not in params:
    params["random_state"] = self.random_state
```

and then:

```python
LocalModelFactory.create(
    name,
    **params,
)
```

Therefore a directly registered custom estimator normally needs to accept:

```python
random_state=None
```

unless you deliberately design another mechanism that accepts arbitrary
keywords.

A constructor such as:

```python
def __init__(
    self,
    random_state=None,
):
    ...
```

is the simplest compatible pattern.

---

## What happens if `random_state` is missing?

This class:

```python
@LocalModelFactory.register(
    "bad_regressor",
    categories={"regression"},
)
class BadRegressor(BaseLocalModel):

    def __init__(self):
        ...
```

can be created by the factory directly without extra parameters, but F-Step
attempts to call it with:

```python
BadRegressor(
    random_state=...
)
```

because `random_state` is injected before factory creation.

Since `BaseFactory.create()` forwards keyword arguments directly to the
registered target, normal Python constructor validation applies.

The result can be:

```text
TypeError:
BadRegressor.__init__() got an unexpected keyword argument 'random_state'
```

!!! important

    Directly registered custom estimators do **not** receive the unsupported
    keyword filtering used by `SklearnLocalModel`.

---

# Custom regressor with parameters

You can expose arbitrary constructor parameters and pass them through
`local_model_params`.

```python
import numpy as np

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "shrunk_mean_regressor",
    categories={"regression"},
)
class ShrunkMeanRegressor(BaseLocalModel):

    def __init__(
        self,
        shrinkage=0.5,
        random_state=None,
    ):
        self.shrinkage = shrinkage
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
    ):
        self.mean_ = float(
            np.mean(y)
        )

        return self

    def predict(
        self,
        X,
    ):
        return np.full(
            len(X),
            self.shrinkage * self.mean_,
            dtype=float,
        )
```

Use it as:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="shrunk_mean_regressor",
    local_model_params={
        "shrinkage": 0.8,
    },
    combiner="gradientcobra",
    random_state=42,
)
```

F-Step creates every cluster-local instance with the equivalent of:

```python
ShrunkMeanRegressor(
    shrinkage=0.8,
    random_state=42,
)
```

unless `random_state` is already included in `local_model_params`.

---

# Custom classifier

A custom classifier follows the same base interface.

```python
import numpy as np
from collections import Counter

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "local_majority_classifier",
    categories={"classification"},
)
class LocalMajorityClassifier(BaseLocalModel):

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
    ):
        counts = Counter(y)

        self.class_ = counts.most_common(
            1
        )[0][0]

        return self

    def predict(
        self,
        X,
    ):
        return np.full(
            len(X),
            self.class_,
            dtype=object,
        )
```

Then:

```python
from kfc_procedure import KFCClassifier

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="local_majority_classifier",
    combiner="combined_classifier",
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

y_pred = model.predict(
    X_test
)
```

---

# Task categories

The factory uses categories to determine whether an estimator is valid for the
current F-Step task.

A regression model should register with:

```python
categories={"regression"}
```

A classifier should register with:

```python
categories={"classification"}
```

The base factory also supports multiple categories.

For example:

```python
@LocalModelFactory.register(
    "my_multitask_model",
    categories={
        "regression",
        "classification",
    },
)
class MyMultitaskModel(BaseLocalModel):
    ...
```

Then:

```python
LocalModelFactory.supports(
    "my_multitask_model",
    "regression",
)
```

and:

```python
LocalModelFactory.supports(
    "my_multitask_model",
    "classification",
)
```

both return `True`.

---

# How F-Step validates a custom model

For string-based models, F-Step checks:

```python
LocalModelFactory.contains(
    name
)
```

then:

```python
LocalModelFactory.supports(
    name,
    self.task,
)
```

before creating the estimator.

Therefore a custom regressor registered only under:

```text
regression
```

cannot accidentally be selected by `KFCClassifier`.

---

# Check your registration

After registering:

```python
print(
    LocalModelFactory.contains(
        "median_regressor"
    )
)
```

Expected:

```text
True
```

Check task support:

```python
print(
    LocalModelFactory.supports(
        "median_regressor",
        "regression",
    )
)
```

Expected:

```text
True
```

And:

```python
print(
    LocalModelFactory.supports(
        "median_regressor",
        "classification",
    )
)
```

Expected:

```text
False
```

---

# Inspect registration metadata

Use:

```python
print(
    LocalModelFactory.info(
        "median_regressor"
    )
)
```

The factory returns a dictionary containing:

```text
name
class
module
categories
metadata
```

---

# Multiple aliases

A custom model can be registered under several names.

```python
@LocalModelFactory.register(
    "median_regressor",
    "median_local",
    categories={"regression"},
)
class MedianRegressor(BaseLocalModel):
    ...
```

Both names point to the same registered target.

You can inspect aliases with:

```python
LocalModelFactory.find_by_class(
    MedianRegressor
)
```

---

# Name normalization

`BaseFactory` normalizes registration names by:

1. requiring strings;
2. stripping surrounding whitespace;
3. converting to lowercase;
4. rejecting empty names.

So:

```python
@LocalModelFactory.register(
    "  My_Model  ",
    categories={"regression"},
)
```

is stored as:

```text
my_model
```

Factory lookups are normalized in the same way.

---

# Duplicate registration protection

The factory rejects duplicate names.

For example, if:

```text
median_regressor
```

already exists, registering another class under the same normalized name raises
a `KeyError`.

The factory also rejects duplicate names within one registration call.

For example:

```python
@LocalModelFactory.register(
    "model",
    "MODEL",
)
```

normalizes both names to the same key and raises a `ValueError`.

---

# Custom metadata

The registry accepts arbitrary metadata in addition to categories.

For example:

```python
@LocalModelFactory.register(
    "median_regressor",
    categories={"regression"},
    author="example",
    version="1.0",
)
class MedianRegressor(BaseLocalModel):
    ...
```

Inspect it with:

```python
LocalModelFactory.info(
    "median_regressor"
)
```

The custom values appear under:

```text
metadata
```

F-Step itself does not currently use arbitrary registration metadata.

---

# One instance per cluster

This is one of the main advantages of string registration.

During F-Step fitting, the code runs:

```python
for div_name, cluster_ids in clusters.items():
    for k in np.unique(cluster_ids):
        model = self._resolve()
        model.fit(
            Xc,
            yc,
        )
```

For a string-based model, `_resolve()` calls:

```python
LocalModelFactory.create(...)
```

each time.

Therefore each populated divergence/cluster pair receives a fresh estimator
instance.

For example:

```text
Euclidean / cluster 0 -> MedianRegressor instance A
Euclidean / cluster 1 -> MedianRegressor instance B
Euclidean / cluster 2 -> MedianRegressor instance C

GKL / cluster 0       -> MedianRegressor instance D
GKL / cluster 1       -> MedianRegressor instance E
GKL / cluster 2       -> MedianRegressor instance F
```

Each object is independently fitted.

---

# Inspect fitted custom models

After KFC fitting:

```python
for divergence, models in model.fstep_.models_.items():
    print(
        divergence
    )

    for key, metadata in models.items():
        print(
            key,
            metadata["cluster"],
            metadata["model"],
        )
```

For the registered custom regressor, each:

```python
metadata["model"]
```

is one fitted custom estimator instance.

---

# Scikit-learn-compatible parameter style

Because `BaseLocalModel` inherits from:

```python
sklearn.base.BaseEstimator
```

custom estimators work best when constructor parameters are stored as
attributes with matching names.

For example:

```python
class MyRegressor(BaseLocalModel):

    def __init__(
        self,
        scale=1.0,
        random_state=None,
    ):
        self.scale = scale
        self.random_state = random_state
```

This follows scikit-learn's estimator conventions and allows inherited
parameter introspection behavior to work correctly.

Avoid performing expensive training work in `__init__()`.

Do training in:

```python
fit()
```

instead.

---

# Fitted attributes

A useful convention is to use trailing underscores for values learned during
`fit()`.

For example:

```python
class MedianRegressor(BaseLocalModel):

    def fit(
        self,
        X,
        y,
    ):
        self.median_ = float(
            np.median(y)
        )

        return self
```

This matches the fitted-attribute style already used throughout the package:

```text
models_
cluster_centers_
labels_
inertia_
strategy_
```

---

# Return `self` from `fit()`

The built-in local models return:

```python
self
```

from `fit()`.

Custom models should follow the same pattern:

```python
def fit(
    self,
    X,
    y,
):
    ...
    return self
```

This matches scikit-learn estimator conventions and the package's local-model
API.

---

# Prediction shape

The F-Step expects each local model's:

```python
predict(X)
```

to return one prediction per input row.

For input:

```text
X.shape == (n_samples, n_features)
```

the normal prediction shape is:

```text
(n_samples,)
```

F-Step assigns the result into a one-dimensional prediction vector:

```python
pred[idx] = model.predict(
    X[idx]
)
```

Therefore custom `predict()` methods should return a one-dimensional vector
compatible with that assignment.

---

# Regression prediction dtype

The F-Step initializes predictions with:

```python
np.full(
    X.shape[0],
    np.nan,
)
```

which produces a floating-point array.

This works naturally for numeric regression predictions.

A custom regressor should normally return numeric values.

---

# Classification prediction dtype caveat

The same F-Step code initializes classification prediction arrays with:

```python
np.full(
    X.shape[0],
    np.nan,
)
```

which is initially floating-point.

If a custom classifier returns non-numeric string labels, assigning them into
that array can fail because the target array is numeric.

For example:

```text
"class_a"
"class_b"
```

may not be assignable into the current float prediction buffer.

!!! warning "Current source limitation"

    The current F-Step prediction buffer is not explicitly created with
    `dtype=object` for classification.

    Numeric class labels are the safest choice with the current F-Step
    implementation.

    If you need arbitrary string labels, F-Step should be updated to allocate
    an object-compatible output buffer for classification.

---

# Custom `predict_proba()`

`BaseLocalModel` defines:

```python
predict_proba(
    X
)
```

but its default implementation raises:

```python
NotImplementedError
```

A custom classifier can override it.

Example:

```python
class MyClassifier(BaseLocalModel):

    def fit(
        self,
        X,
        y,
    ):
        self.classes_ = np.unique(y)
        ...
        return self

    def predict(
        self,
        X,
    ):
        ...

    def predict_proba(
        self,
        X,
    ):
        ...
```

The expected probability format should follow the underlying classifier's own
convention, typically:

```text
(n_samples, n_classes)
```

---

## Current KFC probability limitation

Even if a custom local classifier implements:

```python
predict_proba()
```

the current `FStep` class itself defines only:

```python
fit()
predict()
```

and does not define:

```python
predict_proba()
```

Therefore `KFCClassifier.predict_proba()` cannot currently obtain a
probability matrix from F-Step.

Implementing `predict_proba()` on a custom local model alone is not sufficient
to make end-to-end KFC probabilities work in the current source version.

---

# Validation inside a custom model

The base interface does not automatically validate:

```text
X shape
y shape
finite values
target type
minimum sample count
```

for a custom estimator.

A custom model should perform whatever validation it requires in `fit()` and
`predict()`.

For example:

```python
def fit(
    self,
    X,
    y,
):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(
            "X must be 2-dimensional"
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be 1-dimensional"
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y have incompatible lengths"
        )

    ...
    return self
```

---

# F-Step wraps `ValueError`

If a custom model raises:

```python
ValueError
```

inside its `fit()` method, F-Step catches it and raises a new message:

```text
[FSTEP ERROR] divergence='...', cluster=... failed.
Reason: ...
Hint: cluster contains invalid label distribution.
```

The original exception is preserved as the cause.

Therefore use informative `ValueError` messages in custom estimators.

---

## Example custom validation error

```python
def fit(
    self,
    X,
    y,
):
    if len(y) < 3:
        raise ValueError(
            "At least 3 cluster samples are required."
        )

    ...
```

F-Step will report the relevant divergence and cluster along with that reason.

---

# Option 2 — Pass an estimator object directly

F-Step accepts:

```python
local_model=<object>
```

when the object is not a string.

For example:

```python
custom = MedianRegressor(
    random_state=42
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model=custom,
    combiner="gradientcobra",
)
```

However, the current `_resolve()` implementation simply returns:

```python
self.local_model
```

for every cluster.

It does **not** clone the object.

---

# Important object-reuse behavior

With a direct object:

```text
cluster 0 -> custom.fit(...)
cluster 1 -> same custom.fit(...)
cluster 2 -> same custom.fit(...)
```

The metadata for multiple clusters can therefore reference the same object
after repeated fitting.

The final object state corresponds to the latest fit.

!!! warning "Prefer registry-based custom models"

    For custom local estimators that need independent fitted state per cluster,
    register the estimator and pass its string name.

    Direct model objects are reused, not cloned, in the current F-Step
    implementation.

---

# If you need to wrap an existing estimator

Suppose you already have a third-party estimator class with:

```python
fit()
predict()
```

but it does not inherit `BaseLocalModel`.

A straightforward approach is to write a thin adapter.

```python
from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "external_regressor",
    categories={"regression"},
)
class ExternalRegressorAdapter(
    BaseLocalModel
):

    def __init__(
        self,
        alpha=1.0,
        random_state=None,
    ):
        self.alpha = alpha
        self.random_state = random_state

        self.model = ExternalRegressor(
            alpha=alpha,
            random_state=random_state,
        )

    def fit(
        self,
        X,
        y,
    ):
        self.model.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X,
    ):
        return self.model.predict(
            X
        )
```

This makes the external estimator compatible with the F-Step registry and
cluster-specific instantiation.

---

# Wrapping a scikit-learn estimator manually

Normally this is unnecessary because the package auto-registers scikit-learn
regressors and classifiers.

If you need a specialized wrapper, you can still create one.

```python
from sklearn.linear_model import Ridge

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "my_ridge",
    categories={"regression"},
)
class MyRidge(BaseLocalModel):

    def __init__(
        self,
        alpha=1.0,
        random_state=None,
    ):
        self.alpha = alpha
        self.random_state = random_state

        self.model = Ridge(
            alpha=alpha,
        )

    def fit(
        self,
        X,
        y,
    ):
        self.model.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X,
    ):
        return self.model.predict(
            X
        )
```

---

# When manual wrapping is useful

A custom wrapper is useful when you need behavior that the generic
`SklearnLocalModel` adapter does not provide, such as:

```text
special preprocessing
custom validation
custom prediction transformation
extra fitted diagnostics
third-party estimator integration
special constructor mapping
```

---

# Registering builder functions

The factory's registration decorator stores the decorated target and
`BaseFactory.create()` later calls it with constructor arguments.

The package itself uses this pattern for dynamic scikit-learn registration:

```python
def builder(
    model_cls=cls,
    **kwargs,
):
    return SklearnLocalModel(
        model_cls,
        **kwargs,
    )

LocalModelFactory.register(
    key,
    categories={category},
)(builder)
```

This means a registered target does not have to be a conventional class in
practice; it must be callable with the arguments supplied by
`BaseFactory.create()`.

For ordinary custom extensions, registering a class is simpler and clearer.

---

# Inspect all custom models

You can list the complete registry:

```python
print(
    LocalModelFactory.available()
)
```

Or only regression models:

```python
print(
    LocalModelFactory.available_by_category(
        "regression"
    )
)
```

Or classifiers:

```python
print(
    LocalModelFactory.available_by_category(
        "classification"
    )
)
```

Your custom registration should appear in the relevant list.

---

# Factory lifecycle helpers

Because `LocalModelFactory` inherits `BaseFactory`, it also exposes registry
management methods.

These include:

```python
unregister(name)
clear()
```

For example:

```python
LocalModelFactory.unregister(
    "median_regressor"
)
```

removes only that registration name.

!!! warning

    `clear()` removes **all** entries from the current `LocalModelFactory`
    registry.

    This includes dynamically registered scikit-learn models and package
    built-ins in the current process.

    It is mainly useful for controlled testing, not ordinary application use.

---

# Re-registering after `clear()`

The package's scikit-learn registration is triggered when:

```python
kfc_procedure.core.ml
```

is imported.

If you manually clear the registry later, simply importing an already imported
module again does not necessarily rerun the module-level registration code.

For tests that manipulate the registry, explicitly manage restoration rather
than relying on normal import caching.

---

# Testing a custom estimator before KFC

A useful development pattern is to test the factory and estimator directly.

```python
local = LocalModelFactory.create(
    "median_regressor",
    random_state=42,
)

local.fit(
    X_train,
    y_train,
)

prediction = local.predict(
    X_test
)

print(
    prediction.shape
)
```

Only after that succeeds should you insert it into the full KFC pipeline.

---

# Test task metadata

```python
assert LocalModelFactory.contains(
    "median_regressor"
)

assert LocalModelFactory.supports(
    "median_regressor",
    "regression",
)

assert not LocalModelFactory.supports(
    "median_regressor",
    "classification",
)
```

---

# Test independent F-Step instances

To verify that string registration creates separate cluster models:

```python
models = model.fstep_.models_[
    "euclidean"
]

instances = [
    metadata["model"]
    for metadata
    in models.values()
]

print(
    [
        id(instance)
        for instance
        in instances
    ]
)
```

For a string-registered custom model, the IDs should normally be distinct.

---

# Custom estimator with learned coefficients

A more realistic example:

```python
import numpy as np

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "least_squares_local",
    categories={"regression"},
)
class LeastSquaresLocal(
    BaseLocalModel
):

    def __init__(
        self,
        fit_intercept=True,
        random_state=None,
    ):
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        if self.fit_intercept:
            X_design = np.column_stack(
                [
                    np.ones(len(X)),
                    X,
                ]
            )
        else:
            X_design = X

        coef, *_ = np.linalg.lstsq(
            X_design,
            y,
            rcond=None,
        )

        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef

        return self

    def predict(
        self,
        X,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        return (
            X @ self.coef_
            + self.intercept_
        )
```

Then:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="least_squares_local",
    local_model_params={
        "fit_intercept": True,
    },
    combiner="gradientcobra",
    n_clusters=3,
    random_state=42,
)
```

---

# Inspect custom fitted attributes

After fitting:

```python
local = model.fstep_.models_[
    "euclidean"
]["m0"]["model"]

print(
    local.coef_
)

print(
    local.intercept_
)
```

Because KFC stores the actual fitted local object, all custom fitted attributes
remain accessible.

---

# Custom classifier with probabilities

A simple illustration:

```python
import numpy as np

from kfc_procedure.core.ml import (
    BaseLocalModel,
    LocalModelFactory,
)


@LocalModelFactory.register(
    "prior_classifier",
    categories={"classification"},
)
class PriorClassifier(
    BaseLocalModel
):

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
    ):
        self.classes_, counts = np.unique(
            y,
            return_counts=True,
        )

        self.probabilities_ = (
            counts
            / counts.sum()
        )

        self.majority_ = self.classes_[
            np.argmax(
                self.probabilities_
            )
        ]

        return self

    def predict(
        self,
        X,
    ):
        return np.full(
            len(X),
            self.majority_,
        )

    def predict_proba(
        self,
        X,
    ):
        return np.tile(
            self.probabilities_,
            (
                len(X),
                1,
            ),
        )
```

This local estimator itself supports probabilities.

However, as noted earlier, the current F-Step does not yet aggregate local
probabilities.

---

# Designing for small clusters

Custom estimators should account for the fact that F-Step trains on local
subsets.

A cluster may contain very few observations.

Possible strategies include:

```text
validate minimum sample count
fall back to a simple constant predictor
regularize aggressively
support one-class classification where meaningful
raise an informative ValueError
```

The package does not impose one fallback policy for custom estimators.

---

## Example minimum-size check

```python
def fit(
    self,
    X,
    y,
):
    if len(y) < 5:
        raise ValueError(
            "This local model requires at least 5 samples."
        )

    ...
```

F-Step will add divergence and cluster context to the error.

---

# Do not depend on cluster ID in the estimator constructor

The current F-Step does not pass:

```text
divergence name
cluster ID
```

into `_resolve()`.

Those values are only stored in the F-Step metadata after fitting.

So a registered custom estimator constructor receives only:

```text
local_model_params
random_state
```

unless you explicitly include other keys in `local_model_params`.

If an estimator needs the current cluster ID or divergence name, F-Step itself
must be extended to pass that context.

---

# Do not depend on full training data

A custom local model's `fit()` receives only:

```python
Xc
yc
```

for one cluster.

It does not automatically receive:

```text
the full X
the full y
other cluster data
K-Step centroids
divergence objects
C-Step state
```

This is intentional separation between KFC stages.

---

# Current custom-estimator limitations

| Capability | Current behavior |
| --- | --- |
| registered custom class | supported |
| custom regression category | supported |
| custom classification category | supported |
| multiple aliases | supported |
| custom registration metadata | supported |
| independent model per cluster by string name | supported |
| custom constructor parameters | supported |
| top-level `random_state` | injected automatically |
| unsupported custom kwargs filtered | No |
| direct object cloning | No |
| different custom model per divergence | No |
| different custom model per cluster | No |
| fit-time kwargs | No |
| cluster ID passed to constructor | No |
| divergence passed to constructor | No |
| F-Step probability aggregation | No |
| arbitrary string-label-safe prediction buffer | not guaranteed in current F-Step |

---

# Recommended extension pattern

For the current source version, the most robust pattern is:

```python
@LocalModelFactory.register(
    "my_estimator",
    categories={"regression"},
)
class MyEstimator(BaseLocalModel):

    def __init__(
        self,
        ...,
        random_state=None,
    ):
        # store constructor arguments
        ...

    def fit(
        self,
        X,
        y,
    ):
        # validate and learn fitted state
        ...
        return self

    def predict(
        self,
        X,
    ):
        # return one value per row
        ...
```

Then use:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="my_estimator",
    local_model_params={
        # constructor parameters
    },
    combiner="gradientcobra",
    random_state=42,
)
```

This ensures F-Step creates a fresh estimator for each cluster.

---

# Debugging a custom estimator

## 1. Check registration

```python
print(
    LocalModelFactory.contains(
        "my_estimator"
    )
)
```

---

## 2. Check category

```python
print(
    LocalModelFactory.supports(
        "my_estimator",
        "regression",
    )
)
```

---

## 3. Create it directly

```python
local = LocalModelFactory.create(
    "my_estimator",
    random_state=42,
)
```

If this fails, fix the constructor before testing KFC.

---

## 4. Fit directly

```python
local.fit(
    X_train,
    y_train,
)
```

---

## 5. Check prediction shape

```python
pred = local.predict(
    X_test
)

print(
    np.asarray(
        pred
    ).shape
)
```

Expected:

```text
(n_samples,)
```

---

## 6. Fit KFC

```python
model.fit(
    X_train,
    y_train,
)
```

If a cluster fit fails, inspect the `Reason:` field inside the F-Step error.

---

## 7. Inspect each fitted instance

```python
for divergence, models in model.fstep_.models_.items():
    for name, metadata in models.items():
        print(
            divergence,
            name,
            metadata["cluster"],
            metadata["model"],
        )
```

---

# Quick reference

| Component | Purpose |
| --- | --- |
| `BaseLocalModel` | custom local-estimator interface |
| `LocalModelFactory.register()` | register estimator names and categories |
| `LocalModelFactory.create()` | create new registered instances |
| `LocalModelFactory.supports()` | validate task category |
| `LocalModelFactory.info()` | inspect registration |
| `FStep._resolve()` | creates string-based models or reuses object models |
| `local_model_params` | constructor parameters |
| `random_state` | injected by F-Step when absent |

---

# Mental model

!!! quote ""

    **A custom F-Step estimator should be a small, independent supervised model
    that can be constructed repeatedly and fitted on one cluster at a time.**

\[
\boxed{
\text{register class}
\rightarrow
\text{factory creates one model per cluster}
\rightarrow
\text{fit on local subset}
\rightarrow
\text{predict one F-Step column}
}
\]

