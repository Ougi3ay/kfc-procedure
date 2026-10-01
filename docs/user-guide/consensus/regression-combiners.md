# Regression Combiners

Regression combiners are the C-Step strategies that turn the F-Step prediction
matrix into one final numeric prediction.

The C-Step receives:

\[
P \in \mathbb{R}^{n \times M},
\]

where each row is one sample and each column is the prediction produced by one
divergence-specific F-Step model family.

Conceptually:

```text
F-Step predictions
        │
        ▼
┌──────────────────────────────┐
│ euclidean  gkl  logistic ... │
├──────────────────────────────┤
│   12.4    12.8    12.1      │
│   20.7    21.0    20.4      │
│    8.3     8.1     8.5      │
└──────────────────────────────┘
        │
        ▼
 regression combiner
        │
        ▼
 final prediction
```

The current source registers five regression combiners:

| Registry name | Implementation |
| --- | --- |
| `mean` | `MeanCombiner` |
| `weighted_mean` | `WeightedMeanCombiner` |
| `stacking_regressor` | `StackingRegressorCombiner` |
| `gradientcobra` | `GradientCOBRACombiner` |
| `mixcobra` | `MixCOBRACombiner` |

---

## Choosing a regression combiner

The five strategies fall into three broad groups.

<div class="grid cards" markdown>

-   :material-equal:{ .lg .middle } **Fixed averaging**

    ---

    `mean`

    Uses the same weight for every prediction column.

-   :material-weight:{ .lg .middle } **Learned meta-models**

    ---

    `weighted_mean`

    `stacking_regressor`

    Learn how F-Step prediction columns map to the target.

-   :material-chart-bell-curve:{ .lg .middle } **COBRA-style aggregation**

    ---

    `gradientcobra`

    `mixcobra`

    Use the F-Step prediction matrix as precomputed prediction-space input.

</div>

---

# `mean`

`MeanCombiner` is the simplest regression consensus strategy.

It is registered as:

```text
mean
```

and computes the arithmetic mean across prediction columns.

For one sample:

\[
\widehat y
=
\frac{1}{M}
\sum_{m=1}^{M}p_m.
\]

The implementation is:

```python
return np.mean(
    X,
    axis=1,
)
```

---

## Fit behavior

`MeanCombiner` is stateless.

Its `fit()` method simply returns:

```python
self
```

and does not use the target vector.

```python
def fit(
    self,
    X,
    y=None,
):
    return self
```

---

## Input validation

`combine()` converts the input with:

```python
np.asarray(X)
```

and requires a two-dimensional matrix.

If:

```python
X.ndim != 2
```

it raises:

```text
ValueError: Expected 2D array, got ...
```

---

## Direct example

```python
import numpy as np

from kfc_procedure.core.combiner.regression import (
    MeanCombiner,
)

P = np.array([
    [10.0, 12.0, 11.0],
    [20.0, 19.0, 22.0],
])

combiner = MeanCombiner()

combiner.fit(P)

prediction = combiner.predict(P)

print(prediction)
```

Conceptually:

```text
[11.0, 20.333...]
```

---

## KFC example

```python
from kfc_procedure import KFCRegressor
from kfc_procedure.core.combiner.regression import (
    MeanCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=MeanCombiner(),
    random_state=42,
)
```

Using an instance is important in the current source because of the C-Step
`random_state` forwarding issue described later on this page.

---

# `weighted_mean`

`WeightedMeanCombiner` is registered as:

```text
weighted_mean
```

Despite the name, the current implementation is not a constrained weighted
average.

It fits:

```python
LinearRegression
```

on the F-Step prediction matrix.

The model is:

\[
y
\approx
Pw.
\]

With an intercept enabled:

\[
y
\approx
Pw+b.
\]

---

## Constructor

```python
WeightedMeanCombiner(
    fit_intercept=False,
)
```

The only constructor parameter is:

```text
fit_intercept
```

with default:

```python
False
```

Internally:

```python
self.model = LinearRegression(
    fit_intercept=fit_intercept
)
```

---

## Fit behavior

The input is converted using:

```python
np.asarray(X)
```

and must be two-dimensional.

Then:

```python
self.model.fit(
    X,
    y,
)
```

is called.

---

## Prediction

`combine()` delegates directly to:

```python
self.model.predict(X)
```

---

## Important interpretation

The learned coefficients are not constrained.

The implementation does **not** enforce:

\[
w_m \geq 0
\]

and does **not** enforce:

\[
\sum_m w_m=1.
\]

So the learned model can use:

```text
negative coefficients
coefficients greater than one
coefficients whose sum is not one
```

This is ordinary least-squares combination.

---

## Direct example

```python
from kfc_procedure.core.combiner.regression import (
    WeightedMeanCombiner,
)

combiner = WeightedMeanCombiner(
    fit_intercept=False,
)

combiner.fit(
    P_train,
    y_train,
)

y_pred = combiner.predict(
    P_test
)
```

---

## Inspect learned coefficients

```python
print(
    combiner.model.coef_
)
```

and:

```python
print(
    combiner.model.intercept_
)
```

If F-Step columns are ordered as:

```text
euclidean
gkl
logistic
is
```

then the coefficient vector follows the same matrix-column order.

---

## KFC example

```python
from kfc_procedure import KFCRegressor
from kfc_procedure.core.combiner.regression import (
    WeightedMeanCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=WeightedMeanCombiner(
        fit_intercept=False,
    ),
    random_state=42,
)
```

---

# `stacking_regressor`

`StackingRegressorCombiner` is registered as:

```text
stacking_regressor
```

It learns a second-level regression model from the F-Step prediction matrix.

Conceptually:

\[
P
\rightarrow
g_{\text{meta}}
\rightarrow
\widehat y.
\]

---

## Default meta-model

If no model is supplied, the constructor uses:

```python
LinearRegression()
```

The constructor is:

```python
StackingRegressorCombiner(
    meta_model=None,
)
```

with:

```python
self.meta_model = (
    meta_model
    or LinearRegression()
)
```

---

## Fit behavior

Before fitting, the configured meta-model is cloned:

```python
self.meta_model_ = clone(
    self.meta_model
)
```

Then:

```python
self.meta_model_.fit(
    X,
    y,
)
```

is called.

The fitted clone is stored as:

```python
meta_model_
```

---

## Why clone?

The source deliberately keeps:

```python
meta_model
```

as the configuration object and:

```python
meta_model_
```

as the fitted object.

This follows standard scikit-learn fitted-state conventions.

---

## Fitted-state check

The implementation also stores:

```python
self._is_fitted = True
```

after training.

Calling `combine()` before fitting raises:

```text
RuntimeError:
StackingRegressorCombiner is not fitted.
```

---

## Direct example

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

---

## Custom meta-regressor

```python
from sklearn.ensemble import RandomForestRegressor

combiner = StackingRegressorCombiner(
    meta_model=RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
)
```

The fitted version is then available as:

```python
combiner.meta_model_
```

---

## KFC example

```python
from kfc_procedure import KFCRegressor
from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=StackingRegressorCombiner(),
    random_state=42,
)
```

---

# `gradientcobra`

`GradientCOBRACombiner` is registered as:

```text
gradientcobra
```

It wraps the package's:

```python
GradientCOBRA
```

estimator.

The constructor is:

```python
GradientCOBRACombiner(
    **cobra_params
)
```

and internally creates:

```python
self.cobra = GradientCOBRA(
    **cobra_params
)
```

---

## KFC integration

The important implementation detail is the fit call:

```python
self.cobra.fit(
    X,
    y,
    as_predictions=True,
)
```

Here `X` is the F-Step prediction matrix.

Therefore GradientCOBRA does not fit a new set of base estimators when used as
a C-Step wrapper.

It receives the divergence-level F-Step predictions directly as its
prediction-space representation.

---

## Prediction

The wrapper delegates to:

```python
self.cobra.predict(X)
```

---

## Example

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="gradientcobra",
    combiner_params={
        "distance": "euclidean",
        "kernel": "rbf",
        "n_cv": 5,
    },
    random_state=42,
)
```

Because the wrapper accepts:

```python
**cobra_params
```

the `random_state` keyword injected by C-Step is accepted and forwarded to
`GradientCOBRA`.

---

## Inspect the fitted wrapper

After fitting:

```python
strategy = model.cstep_.strategy_
```

The wrapped estimator is:

```python
strategy.cobra
```

For example:

```python
print(
    strategy.cobra
)
```

Fitted GradientCOBRA attributes depend on the selected configuration.

For the algorithm and bandwidth optimization behavior, see:

[GradientCOBRA](../../getting-started/concepts/gradientcobra.md)

---

# `mixcobra`

`MixCOBRACombiner` is registered as:

```text
mixcobra
```

It wraps:

```python
MixCOBRARegressor
```

The constructor is:

```python
MixCOBRACombiner(
    **cobra_params
)
```

and creates:

```python
self.cobra = MixCOBRARegressor(
    **cobra_params
)
```

---

## KFC integration

The wrapper fits MixCOBRA using:

```python
self.cobra.fit(
    X,
    y,
    as_predictions=True,
)
```

Again, `X` is the F-Step prediction matrix.

This is a specific KFC integration path.

---

## Important distinction from standalone MixCOBRA

Standalone MixCOBRA is organized around input-space and prediction-space
information.

Inside the KFC C-Step wrapper, the F-Step prediction matrix is supplied as
precomputed predictions using:

```python
as_predictions=True
```

So this wrapper should be understood as KFC feeding its divergence-level
prediction representation into MixCOBRA's precomputed-prediction mode.

For the standalone algorithm, see:

[MixCOBRA](../../getting-started/concepts/mixcobra.md)

---

## Example

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mixcobra",
    combiner_params={
        "distance": "euclidean",
        "kernel": "rbf",
        "n_cv": 5,
    },
    random_state=42,
)
```

---

## Inspect the fitted wrapper

```python
strategy = model.cstep_.strategy_

cobra = strategy.cobra

print(
    cobra
)
```

The wrapper itself adds no additional learned state beyond the underlying
`MixCOBRARegressor`.

---

# Comparison

| Combiner | Uses target during `fit()` | Learned object | Main operation |
| --- | :---: | --- | --- |
| `mean` | No | none | arithmetic mean |
| `weighted_mean` | Yes | `LinearRegression` | linear coefficients |
| `stacking_regressor` | Yes | cloned meta-regressor | second-level regression |
| `gradientcobra` | Yes | `GradientCOBRA` | prediction-space kernel aggregation |
| `mixcobra` | Yes | `MixCOBRARegressor` | MixCOBRA precomputed-prediction aggregation |

---

# Current C-Step `random_state` issue

The current C-Step builds string-based combiners with:

```python
params = dict(
    self.combiner_params
)

if "random_state" not in params:
    params["random_state"] = self.random_state
```

and then:

```python
CombinerFactory.create(
    name,
    **params
)
```

This means every string-based combiner receives a `random_state` keyword.

---

## Affected regression combiners

These constructors do **not** accept `random_state`:

```python
MeanCombiner()
```

```python
WeightedMeanCombiner(
    fit_intercept=False
)
```

```python
StackingRegressorCombiner(
    meta_model=None
)
```

Therefore the current string-based C-Step path can attempt calls equivalent to:

```python
MeanCombiner(
    random_state=42
)
```

or even:

```python
MeanCombiner(
    random_state=None
)
```

and raise:

```text
TypeError:
... got an unexpected keyword argument 'random_state'
```

---

## COBRA wrappers are not affected

These constructors accept arbitrary keyword arguments:

```python
GradientCOBRACombiner(
    **cobra_params
)
```

```python
MixCOBRACombiner(
    **cobra_params
)
```

So:

```text
gradientcobra
mixcobra
```

remain compatible with the current C-Step parameter forwarding design.

---

# Workaround for simple combiners

Pass a pre-instantiated combiner object instead of a string.

## Mean

```python
from kfc_procedure.core.combiner.regression import (
    MeanCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=MeanCombiner(),
    random_state=42,
)
```

---

## Weighted mean

```python
from kfc_procedure.core.combiner.regression import (
    WeightedMeanCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=WeightedMeanCombiner(
        fit_intercept=False,
    ),
    random_state=42,
)
```

---

## Stacking

```python
from kfc_procedure.core.combiner.regression import (
    StackingRegressorCombiner,
)

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=StackingRegressorCombiner(),
    random_state=42,
)
```

For a non-string object, C-Step returns it directly:

```python
if not isinstance(
    self.combiner,
    str,
):
    return self.combiner
```

so the automatic constructor keyword injection is bypassed.

---

# Direct combiner use

The regression combiners can also be used outside the full KFC pipeline.

Suppose:

```python
P_train
```

already contains model predictions.

Then:

```python
combiner.fit(
    P_train,
    y_train,
)
```

followed by:

```python
combiner.predict(
    P_test
)
```

uses the same combiner API as C-Step.

---

# Example prediction matrix

```python
import numpy as np

P_train = np.array([
    [10.1, 10.4, 10.0],
    [12.0, 12.2, 11.8],
    [18.5, 18.0, 18.8],
    [25.1, 24.7, 25.4],
])

y_train = np.array([
    10.2,
    12.1,
    18.4,
    25.0,
])
```

Then any regression combiner can consume:

```text
P_train.shape == (4, 3)
```

where the three columns represent three prediction sources.

---

# Inspect the C-Step prediction matrix

For a fitted KFC regressor:

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

The column names can be recovered with:

```python
divergence_names = list(
    model.fstep_.models_.keys()
)
```

---

# Interpret weighted coefficients

If:

```python
divergence_names = [
    "euclidean",
    "gkl",
    "logistic",
]
```

and:

```python
weights = np.array([
    0.5,
    -0.2,
    0.9,
])
```

then the learned model corresponds to:

\[
\widehat y
=
0.5p_{\text{euclidean}}
-
0.2p_{\text{gkl}}
+
0.9p_{\text{logistic}}
\]

when `fit_intercept=False`.

This illustrates why `weighted_mean` should not be interpreted as a convex
average.

---

# Inspect stacking state

For a fitted stacking strategy:

```python
strategy = model.cstep_.strategy_

print(
    strategy.meta_model_
)
```

If the default `LinearRegression` is used:

```python
print(
    strategy.meta_model_.coef_
)
```

You can compare these coefficients with the F-Step column order.

---

# Inspect COBRA state

For either COBRA regression wrapper:

```python
strategy = model.cstep_.strategy_

cobra = strategy.cobra
```

The exact fitted attributes belong to the underlying COBRA estimator.

For example, depending on the implementation and mode:

```python
print(
    cobra.optimization_outputs_
)
```

may expose optimization information after fitting.

Refer to the dedicated concept pages for algorithm-specific details rather
than assuming both COBRA classes expose identical fitted state.

---

# Input shape requirements

`mean`, `weighted_mean`, and `stacking_regressor` explicitly validate their
training or combination input as a two-dimensional array.

Expected shape:

```text
(n_samples, n_prediction_columns)
```

For KFC:

```text
(n_samples, n_divergences)
```

A one-dimensional input such as:

```text
(n_samples,)
```

is not accepted by these strategies.

---

# One divergence

A KFC model can use a single divergence:

```python
divergences=[
    "euclidean"
]
```

Then the C-Step matrix has shape:

```text
(n_samples, 1)
```

With `mean`, the one input column is returned unchanged numerically.

With learned combiners, the strategy can still fit a transformation from that
single column to the target.

---

# Several divergences

With:

```python
divergences=[
    "euclidean",
    "gkl",
    "logistic",
    "is",
]
```

the C-Step receives:

```text
(n_samples, 4)
```

This allows the combiner to aggregate or learn relationships across the four
divergence-specific predictions.

---

# Learned combiners train on the KFC aggregation subset

`KFCProcedure.fit()` internally separates the data into:

```text
D_k
D_l
```

The K-Step and F-Step are fitted using:

```text
D_k
```

Then `D_l` is passed through the fitted K/F stages to create:

```text
P_l
```

and the C-Step is fitted with:

```python
cstep.fit(
    P_l,
    y_l,
)
```

Therefore learned regression combiners are trained on prediction vectors
generated for the aggregation subset.

---

# Which combiner is simplest?

The source itself does not rank the combiners.

Their behavior differs as follows:

```text
mean
    fixed arithmetic rule

weighted_mean
    fitted linear rule

stacking_regressor
    fitted customizable regression rule

gradientcobra
    fitted GradientCOBRA rule

mixcobra
    fitted MixCOBRA rule
```

Use the strategy that matches the aggregation behavior you want to evaluate.

---

# Custom regression combiner

You can implement a custom strategy by subclassing:

```python
BaseCombiner
```

and registering it under:

```text
regression
```

Example:

```python
import numpy as np

from kfc_procedure.core.combiner.base import (
    BaseCombiner,
)

from kfc_procedure.core.combiner import (
    CombinerFactory,
)


@CombinerFactory.register(
    "median",
    categories={"regression"},
)
class MedianCombiner(
    BaseCombiner
):

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

    def fit(
        self,
        X,
        y=None,
    ):
        return self

    def combine(
        self,
        X,
    ):
        X = np.asarray(X)

        return np.median(
            X,
            axis=1,
        )
```

The `random_state` parameter is included because the current C-Step injects it
for every string-based combiner.

---

# Debugging regression consensus

## Inspect the selected strategy

```python
print(
    type(
        model.cstep_.strategy_
    ).__name__
)
```

---

## Inspect F-Step input

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

print(P)
```

---

## Check for missing values

For numeric regression predictions:

```python
import numpy as np

print(
    np.isfinite(P).all()
)
```

Learned combiners may fail if upstream predictions contain `NaN` or infinity.

---

## Check column order

```python
print(
    list(
        model.fstep_.models_.keys()
    )
)
```

This is especially important when interpreting learned linear coefficients.

---

## Check weighted-mean coefficients

```python
strategy = model.cstep_.strategy_

print(
    strategy.model.coef_
)
```

---

## Check stacking meta-model

```python
print(
    model.cstep_.strategy_.meta_model_
)
```

---

## Check COBRA wrapper

```python
print(
    model.cstep_.strategy_.cobra
)
```

---

# Quick reference

| Name | Constructor | Stateless? | Current string-path compatibility |
| --- | --- | :---: | :---: |
| `mean` | `MeanCombiner()` | Yes | affected by `random_state` injection |
| `weighted_mean` | `WeightedMeanCombiner(fit_intercept=False)` | No | affected |
| `stacking_regressor` | `StackingRegressorCombiner(meta_model=None)` | No | affected |
| `gradientcobra` | `GradientCOBRACombiner(**cobra_params)` | No | compatible |
| `mixcobra` | `MixCOBRACombiner(**cobra_params)` | No | compatible |

---

# Mental model

!!! quote ""

    **Regression consensus answers one question: given several
    divergence-specific numeric predictions, how should they become one final
    number?**

\[
\boxed{
\begin{bmatrix}
p_1(x) &
p_2(x) &
\cdots &
p_M(x)
\end{bmatrix}
\rightarrow
g
\rightarrow
\widehat y(x)
}
\]

The current source provides a fixed arithmetic rule, two learned meta-model
approaches, and two COBRA wrappers.

