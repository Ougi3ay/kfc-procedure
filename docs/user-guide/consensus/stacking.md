# Stacking

**Stacking** is a learned C-Step strategy that uses the F-Step prediction matrix
as input to a second-level predictive model.

The package provides two task-specific stacking combiners:

```text
stacking_regressor
stacking_classifier
```

They are implemented by:

```python
StackingRegressorCombiner
StackingClassifierCombiner
```

Both follow the same structure:

\[
\boxed{
\text{F-Step prediction matrix}
\rightarrow
\text{meta-model}
\rightarrow
\text{final prediction}
}
\]

---

## Core idea

Suppose the F-Step produces predictions from \(M\) divergence-specific local
model families.

For \(n\) observations, the C-Step receives:

\[
P
=
\begin{bmatrix}
p_{1,1} & \cdots & p_{1,M}\\
\vdots & & \vdots\\
p_{n,1} & \cdots & p_{n,M}
\end{bmatrix}.
\]

Stacking treats the columns of \(P\) as features for a new predictive model.

For regression:

\[
\widehat y
=
g_{\text{reg}}(P).
\]

For classification:

\[
\widehat c
=
g_{\text{clf}}(P).
\]

The source calls these second-level models:

```text
meta_model
meta_model_
```

where:

```text
meta_model
    configured estimator

meta_model_
    fitted clone
```

---

## Where stacking appears in KFC

```mermaid
flowchart LR
    K["K-Step"]
    F["F-Step"]
    P["Prediction matrix"]
    M["Meta-model"]
    Y["Final prediction"]

    K --> F --> P --> M --> Y
```

The stacking combiner does not use the original KFC feature matrix directly.

It receives only the F-Step prediction matrix and the target values supplied to
the C-Step.

---

# Regression stacking

The regression implementation is:

```python
StackingRegressorCombiner
```

and is registered as:

```text
stacking_regressor
```

The class documentation describes it as:

> Stacking combiner using a regression meta-model.

Its purpose is to learn a mapping from base predictions to target values.

---

## Constructor

```python
StackingRegressorCombiner(
    meta_model=None,
)
```

If no meta-model is supplied, the current source uses:

```python
LinearRegression()
```

through:

```python
self.meta_model = (
    meta_model
    or LinearRegression()
)
```

It also initializes:

```python
self._is_fitted = False
```

---

## Regression fit flow

The `fit()` method first converts the prediction matrix:

```python
X = np.asarray(X)
```

Then it requires two-dimensional input:

```python
if X.ndim != 2:
    raise ValueError(
        f"Expected 2D array, got {X.shape}"
    )
```

Next it clones the configured meta-model:

```python
self.meta_model_ = clone(
    self.meta_model
)
```

and fits the clone:

```python
self.meta_model_.fit(
    X,
    y,
)
```

Finally:

```python
self._is_fitted = True
```

and the method returns:

```python
self
```

---

## Why the meta-model is cloned

The source does not fit:

```python
self.meta_model
```

directly.

Instead:

```python
meta_model
```

remains the configured template while:

```python
meta_model_
```

becomes the fitted estimator.

This is useful when inspecting configuration versus learned state.

---

## Regression prediction

The `combine()` method first checks:

```python
self._is_fitted
```

If stacking has not been fitted, it raises:

```text
RuntimeError:
StackingRegressorCombiner is not fitted.
```

Otherwise prediction is:

```python
return self.meta_model_.predict(
    np.asarray(X)
)
```

Because `BaseCombiner.predict()` delegates to `combine()`, both:

```python
combiner.combine(P)
```

and:

```python
combiner.predict(P)
```

use the fitted meta-regressor.

---

## Direct regression example

```python
from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

combiner = StackingRegressorCombiner()

combiner.fit(
    P_train,
    y_train,
)

y_pred = combiner.predict(
    P_test
)
```

With the default constructor, the fitted model is a cloned:

```python
LinearRegression
```

instance.

---

## Inspect the fitted regression meta-model

```python
print(
    combiner.meta_model_
)
```

For the default linear regression model:

```python
print(
    combiner.meta_model_.coef_
)

print(
    combiner.meta_model_.intercept_
)
```

The coefficients correspond to the prediction-matrix columns.

---

# Classification stacking

The classification implementation is:

```python
StackingClassifierCombiner
```

and is registered as:

```text
stacking_classifier
```

The source describes it as a stacking-based classification combiner that
learns a mapping from base predictions to final class labels.

---

## Constructor

```python
StackingClassifierCombiner(
    meta_model=None,
)
```

The default meta-classifier is:

```python
LogisticRegression(
    max_iter=1000,
)
```

through:

```python
self.meta_model = (
    meta_model
    or LogisticRegression(
        max_iter=1000
    )
)
```

The source also initializes:

```python
self._is_fitted = False
```

---

## Classification fit flow

Like regression, fitting begins with:

```python
X = np.asarray(X)
```

and validates:

```python
X.ndim == 2
```

Then:

```python
self.meta_model_ = clone(
    self.meta_model
)
```

followed by:

```python
self.meta_model_.fit(
    X,
    y,
)
```

Finally:

```python
self._is_fitted = True
```

---

## Classification prediction

Before prediction:

```python
if not self._is_fitted:
    raise RuntimeError(
        "StackingClassifierCombiner is not fitted."
    )
```

Then:

```python
return self.meta_model_.predict(
    np.asarray(X)
)
```

---

## Direct classification example

```python
from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

combiner = StackingClassifierCombiner()

combiner.fit(
    P_train,
    y_train,
)

y_pred = combiner.predict(
    P_test
)
```

---

## Inspect the fitted classifier

```python
print(
    combiner.meta_model_
)
```

For the default logistic regression classifier:

```python
print(
    combiner.meta_model_.classes_
)

print(
    combiner.meta_model_.coef_
)

print(
    combiner.meta_model_.intercept_
)
```

---

# Prediction matrix shape

Both stacking classes expect:

```text
(n_samples, n_models)
```

In KFC, this means:

```text
(n_samples, n_divergences)
```

For example, if KFC uses:

```python
divergences=[
    "euclidean",
    "gkl",
    "logistic",
    "is",
]
```

the stacking meta-model receives four features per sample.

---

## Regression matrix example

```text
             euclidean      gkl      logistic      is
sample 1        12.4        12.8       12.1       12.5
sample 2        20.7        21.0       20.4       20.8
sample 3         8.3         8.1        8.5        8.2
```

---

## Classification matrix example

```text
             euclidean      gkl      logistic      is
sample 1          0           0           1          0
sample 2          1           1           1          1
sample 3          2           1           2          2
```

The current stacking classes do not transform or encode the prediction matrix
themselves.

They pass it directly to the configured meta-model.

---

# Important classification detail

For classification, divergence-level class predictions are passed directly as
features.

For example:

```text
[0, 0, 1, 0]
```

is presented to the default logistic-regression meta-classifier as a numeric
feature vector.

The stacking implementation does not one-hot encode those divergence-specific
class predictions.

!!! note

    This means the numeric representation of class labels becomes part of the
    meta-classifier input when using the current stacking implementation.

---

# Default regression meta-model

The default regression stacking configuration is equivalent to:

```python
StackingRegressorCombiner(
    meta_model=LinearRegression()
)
```

This learns a linear mapping:

\[
\widehat y
=
\beta_0
+
\sum_{m=1}^{M}
\beta_m p_m.
\]

Unlike `WeightedMeanCombiner`, the default `LinearRegression()` used by
`StackingRegressorCombiner` uses scikit-learn's normal default
`fit_intercept=True`.

---

## Difference from `weighted_mean`

Both default regression stacking and `weighted_mean` use
`LinearRegression`, but their source defaults differ.

`WeightedMeanCombiner` creates:

```python
LinearRegression(
    fit_intercept=False
)
```

by default.

`StackingRegressorCombiner` creates:

```python
LinearRegression()
```

which uses the estimator's default intercept behavior.

So even before custom meta-models are introduced, the two classes are not
identical in their default configuration.

---

# Default classification meta-model

The classification combiner uses:

```python
LogisticRegression(
    max_iter=1000
)
```

when no custom model is supplied.

The stacking source does not add any further default parameters.

---

# Custom regression meta-model

Pass any compatible cloneable regressor through:

```python
meta_model
```

For example:

```python
from sklearn.ensemble import (
    RandomForestRegressor,
)

from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

combiner = StackingRegressorCombiner(
    meta_model=RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
)
```

Then:

```python
combiner.fit(
    P_train,
    y_train,
)
```

clones and fits the configured random forest.

---

# Custom classification meta-model

Likewise:

```python
from sklearn.ensemble import (
    RandomForestClassifier,
)

from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

combiner = StackingClassifierCombiner(
    meta_model=RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
)
```

The source clones the supplied estimator before training it.

---

# Clone compatibility

Both stacking implementations use:

```python
sklearn.base.clone
```

Therefore a custom meta-model must be compatible with scikit-learn cloning.

In practice, it should behave like a scikit-learn estimator and expose its
constructor configuration through the normal estimator parameter interface.

If `clone()` cannot reconstruct the supplied object from its parameters,
stacking fitting will fail before the meta-model's own `fit()` call.

---

# KFC regression usage

Conceptually, the intended string configuration is:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="stacking_regressor",
)
```

However, the current C-Step implementation injects `random_state` into every
string-based combiner constructor.

`StackingRegressorCombiner.__init__()` currently accepts only:

```python
meta_model=None
```

and does not accept:

```python
random_state
```

Therefore the string path is affected by the current constructor mismatch.

---

# Current KFC regression workaround

Pass an instance:

```python
from kfc_procedure import KFCRegressor
from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner=StackingRegressorCombiner(),
    random_state=42,
)
```

Because the combiner is already an object, C-Step returns it directly rather
than reconstructing it through `CombinerFactory`.

---

# KFC classification usage

The intended string name is:

```text
stacking_classifier
```

but the same current C-Step issue applies because:

```python
StackingClassifierCombiner.__init__(
    meta_model=None
)
```

does not accept `random_state`.

Use an instance with the current source version:

```python
from kfc_procedure import KFCClassifier
from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

model = KFCClassifier(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="logistic_regression",
    combiner=StackingClassifierCombiner(),
    random_state=42,
)
```

---

# Custom meta-model in KFC

Regression:

```python
from sklearn.ensemble import (
    RandomForestRegressor,
)

stacker = StackingRegressorCombiner(
    meta_model=RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=stacker,
    random_state=42,
)
```

Classification:

```python
from sklearn.ensemble import (
    RandomForestClassifier,
)

stacker = StackingClassifierCombiner(
    meta_model=RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner=stacker,
    random_state=42,
)
```

---

# How KFC trains stacking

KFC internally separates the supplied training data into two subsets.

Conceptually:

```text
D_k
    │
    ├── K-Step
    └── F-Step
          │
          ▼
     fitted local models

D_l
    │
    └── pass through fitted K-Step and F-Step
          │
          ▼
         P_l
          │
          ▼
      stacking fit
```

The stacking combiner therefore fits on:

```python
P_l
```

and:

```python
y_l
```

rather than on the same rows used to fit the cluster-local F-Step estimators.

---

# Reconstruct stacking input

After fitting a KFC model, create the same kind of prediction representation
for new data:

```python
clusters = model.kstep_.predict(
    X_test
)

P_test = model.fstep_.predict(
    X_test,
    clusters,
)
```

Then the fitted C-Step does:

```python
y_pred = model.cstep_.predict(
    P_test
)
```

---

# Inspect the F-Step column order

The stacking meta-model sees columns in the order returned by F-Step.

Inspect them with:

```python
divergence_names = list(
    model.fstep_.models_.keys()
)

print(
    divergence_names
)
```

This is especially useful for interpreting:

```python
meta_model_.coef_
```

in linear stacking models.

---

# Interpret regression stacking coefficients

Suppose:

```python
divergence_names = [
    "euclidean",
    "gkl",
    "logistic",
]
```

and the fitted default linear meta-model has:

```python
coef_ = [
    0.4,
    -0.1,
    0.8,
]
```

Then the learned stacking rule has the form:

\[
\widehat y
=
\beta_0
+
0.4p_{\text{euclidean}}
-
0.1p_{\text{gkl}}
+
0.8p_{\text{logistic}}.
\]

The coefficients are not constrained by the stacking class.

Their behavior is determined by the selected meta-model.

---

# Inspect classification stacking coefficients

For the default logistic-regression meta-model:

```python
strategy = model.cstep_.strategy_

meta = strategy.meta_model_

print(
    meta.classes_
)

print(
    meta.coef_
)

print(
    meta.intercept_
)
```

Interpretation follows scikit-learn's `LogisticRegression` representation.

The stacking combiner itself adds no additional coefficient transformation.

---

# No stacking `predict_proba()`

The classification stacking implementation does not expose:

```python
predict_proba()
```

even though its default:

```python
LogisticRegression
```

supports probabilities.

The combiner only defines:

```python
fit()
combine()
```

and inherits:

```python
predict()
```

from `BaseCombiner`.

Therefore:

```python
CStep.predict_proba()
```

does not find a `predict_proba` method on `StackingClassifierCombiner`.

---

## Direct access to meta-model probabilities

If you already have a fitted `StackingClassifierCombiner`, you can technically
access the fitted underlying meta-model directly:

```python
strategy = model.cstep_.strategy_

proba = strategy.meta_model_.predict_proba(
    P_test
)
```

when the configured meta-classifier supports that method.

This is direct use of the underlying estimator, not a probability API exposed
by `StackingClassifierCombiner` itself.

---

# Fit-before-predict requirement

Both stacking classes use their own flag:

```python
_is_fitted
```

rather than scikit-learn's `check_is_fitted()` inside `combine()`.

Immediately after construction:

```python
combiner._is_fitted == False
```

After successful fitting:

```python
combiner._is_fitted == True
```

Calling:

```python
combiner.predict(P)
```

too early raises the task-specific `RuntimeError`.

---

# Input validation scope

During `fit()`, both classes explicitly check only:

```text
X is two-dimensional
```

The stacking classes themselves do not add explicit checks for:

```text
X and y length agreement
finite values
target type
minimum sample count
```

Those conditions are left to the selected meta-model or NumPy/scikit-learn
operations.

During `combine()`, the source converts with:

```python
np.asarray(X)
```

but does not explicitly repeat the `ndim == 2` check before calling the fitted
meta-model.

---

# One divergence

If KFC uses one divergence:

```python
divergences=[
    "euclidean"
]
```

then the stacking matrix has shape:

```text
(n_samples, 1)
```

A meta-model can still be trained on this single prediction feature.

For default regression stacking, that means fitting a linear relationship from
one F-Step prediction column to the target.

For default classification stacking, it means fitting logistic regression on
one numeric prediction feature.

---

# Several divergences

With several divergences, stacking can learn interactions only to the extent
that the chosen meta-model supports them.

The default regression meta-model is linear:

```text
LinearRegression
```

The default classification meta-model is also linear in its input features:

```text
LogisticRegression
```

If nonlinear meta-level relationships are needed, supply a different
clone-compatible meta-model.

---

# Stacking is supervised

Unlike:

```text
mean
majority_vote
```

stacking requires target values during `fit()`.

Regression:

```python
combiner.fit(
    P_train,
    y_train,
)
```

Classification:

```python
combiner.fit(
    P_train,
    y_train,
)
```

The target values determine the fitted `meta_model_`.

---

# Stacking vs weighted mean

The current regression implementations differ in structure.

| Property | `weighted_mean` | `stacking_regressor` |
| --- | --- | --- |
| Default learner | `LinearRegression(fit_intercept=False)` | `LinearRegression()` |
| Custom learner | No constructor hook | `meta_model=` |
| Fitted estimator attribute | `model` | `meta_model_` |
| Clone configured estimator | No | Yes |
| Explicit fitted flag | No | Yes |

Both operate on the same kind of prediction matrix, but `stacking_regressor`
is the customizable meta-regression abstraction.

---

# Stacking vs majority vote

For classification:

```text
majority_vote
    no training
    counts row labels

stacking_classifier
    supervised training
    learns mapping from prediction rows to labels
```

Stacking can learn that different divergence columns carry different
information, subject to the selected meta-model.

---

# Stacking vs CombinedClassifier

Both are learned classification C-Step strategies, but the implementations are
different.

`stacking_classifier`:

```text
prediction matrix
    ↓
meta-classifier
    ↓
class label
```

`combined_classifier`:

```text
prediction matrix
    ↓
CombinedClassifier
    ↓
prediction-space distance/kernel consensus
    ↓
class label
```

For the second approach, see:

[CombinedClassifier](../../getting-started/concepts/combined-classifier.md)

---

# Direct regression experiment

```python
import numpy as np

from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

P_train = np.array([
    [10.0, 10.5],
    [12.0, 11.7],
    [18.0, 18.4],
    [25.0, 24.8],
])

y_train = np.array([
    10.2,
    11.9,
    18.3,
    25.1,
])

P_test = np.array([
    [14.0, 14.3],
    [20.0, 19.8],
])

stacker = StackingRegressorCombiner()

stacker.fit(
    P_train,
    y_train,
)

prediction = stacker.predict(
    P_test
)

print(
    prediction
)
```

---

# Direct classification experiment

```python
import numpy as np

from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

P_train = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 0],
    [1, 1, 0],
])

y_train = np.array([
    0,
    1,
    0,
    1,
    0,
    1,
])

P_test = np.array([
    [0, 0, 1],
    [1, 1, 1],
])

stacker = StackingClassifierCombiner()

stacker.fit(
    P_train,
    y_train,
)

prediction = stacker.predict(
    P_test
)

print(
    prediction
)
```

---

# Debugging stacking

## Check the matrix shape

```python
print(
    P_train.shape
)
```

Expected:

```text
(n_samples, n_prediction_columns)
```

---

## Check fitted state

```python
print(
    stacker._is_fitted
)
```

After successful fitting:

```text
True
```

---

## Inspect the fitted meta-model

```python
print(
    stacker.meta_model_
)
```

---

## Inspect KFC column order

```python
print(
    list(
        model.fstep_.models_.keys()
    )
)
```

---

## Reconstruct the matrix

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

print(
    P[:5]
)
```

---

## Check the current string-path error

If KFC raises:

```text
unexpected keyword argument 'random_state'
```

while using:

```text
stacking_regressor
```

or:

```text
stacking_classifier
```

pass a pre-instantiated stacking combiner instead.

---

# Current source limitations

| Area | Current behavior |
| --- | --- |
| regression default meta-model | `LinearRegression()` |
| classification default meta-model | `LogisticRegression(max_iter=1000)` |
| custom meta-model | supported |
| meta-model cloning | supported |
| fit input validation | only explicit 2D `X` check |
| regression probability output | not applicable |
| classification combiner `predict_proba()` | not implemented |
| classification F-Step labels | passed directly as meta-features |
| one-hot encoding of F-Step labels | not implemented |
| string KFC integration | affected by C-Step `random_state` injection |
| combiner instance integration | supported |
| nonlinear stacking | possible through custom meta-model |

---

# Quick reference

| Feature | Regression | Classification |
| --- | --- | --- |
| Registry name | `stacking_regressor` | `stacking_classifier` |
| Class | `StackingRegressorCombiner` | `StackingClassifierCombiner` |
| Default model | `LinearRegression()` | `LogisticRegression(max_iter=1000)` |
| Fitted model | `meta_model_` | `meta_model_` |
| Clones meta-model | Yes | Yes |
| Requires target in `fit()` | Yes | Yes |
| `predict_proba()` on combiner | — | No |
| Current string path affected by `random_state` injection | Yes | Yes |

---

# Mental model

!!! quote ""

    **Stacking treats the F-Step predictions as a new feature space and learns
    a second-level model on top of them.**

\[
\boxed{
P
=
\begin{bmatrix}
p_1 & p_2 & \cdots & p_M
\end{bmatrix}
\rightarrow
\text{meta-model}
\rightarrow
\widehat y
}
\]

The regression and classification implementations differ mainly in their
default meta-models; their fitting structure is otherwise closely parallel.

