# Classification Combiners

Classification combiners are the C-Step strategies that turn the F-Step
prediction matrix into one final class label.

The C-Step receives one prediction column per divergence:

\[
P
=
\begin{bmatrix}
p_{1,1} & \cdots & p_{1,M} \\
\vdots & & \vdots \\
p_{n,1} & \cdots & p_{n,M}
\end{bmatrix},
\]

where each entry is a class prediction produced by one divergence-specific
local-model family.

Conceptually:

```text
F-Step predictions
        │
        ▼
┌──────────────────────────────┐
│ euclidean   gkl   logistic   │
├──────────────────────────────┤
│     0        0        1      │
│     1        1        1      │
│     2        1        2      │
└──────────────────────────────┘
        │
        ▼
classification combiner
        │
        ▼
final class label
```

The current source registers three classification combiners:

| Registry name | Implementation |
| --- | --- |
| `majority_vote` | `MajorityVoteCombiner` |
| `stacking_classifier` | `StackingClassifierCombiner` |
| `combined_classifier` | `CobraClassifierCombiner` |

---

## Classification consensus families

<div class="grid cards" markdown>

-   :material-vote:{ .lg .middle } **Hard voting**

    ---

    `majority_vote`

    Uses the most frequent class label in each prediction row.

-   :material-layers-triple:{ .lg .middle } **Stacking**

    ---

    `stacking_classifier`

    Fits a meta-classifier to the divergence prediction matrix.

-   :material-chart-bell-curve:{ .lg .middle } **Prediction-space consensus**

    ---

    `combined_classifier`

    Wraps `CombinedClassifier` and uses the F-Step matrix directly as
    prediction-space input.

</div>

---

# `majority_vote`

`MajorityVoteCombiner` is registered as:

```text
majority_vote
```

It performs row-wise hard voting.

For one sample:

```text
[1, 1, 0, 1]
```

the final output is:

```text
1
```

---

## Implementation

The current source uses:

```python
Counter(
    row
).most_common(1)[0][0]
```

for every row.

The output array is explicitly allocated as:

```python
np.empty(
    n_samples,
    dtype=object,
)
```

so the combiner itself can return non-numeric class labels.

---

## Fit behavior

`MajorityVoteCombiner` is stateless.

Its `fit()` method is:

```python
def fit(
    self,
    X,
    y=None,
):
    return self
```

It does not learn from the target labels.

---

## Input validation

The prediction matrix is converted with:

```python
np.asarray(X)
```

and must be two-dimensional.

If:

```python
X.ndim != 2
```

the combiner raises:

```text
ValueError:
Expected 2D array, got ...
```

---

## Direct example

```python
import numpy as np

from kfc_procedure.core.combiner.classification import (
    MajorityVoteCombiner,
)

P = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [2, 1, 2],
])

combiner = MajorityVoteCombiner()

combiner.fit(P)

prediction = combiner.predict(P)

print(prediction)
```

Conceptually:

```text
[0, 1, 2]
```

---

## Tie behavior

The current source does not implement a separate tie-breaking policy.

Tie behavior comes from:

```python
Counter(...).most_common(1)
```

and therefore depends on the order in which equally frequent labels are
encountered.

!!! note

    If deterministic domain-specific tie-breaking matters, implement a custom
    combiner rather than assuming a special KFC tie policy.

---

## KFC example

Because of the current C-Step constructor issue, use a combiner instance:

```python
from kfc_procedure import KFCClassifier
from kfc_procedure.core.combiner.classification import (
    MajorityVoteCombiner,
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner=MajorityVoteCombiner(),
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

The reason for preferring an instance in the current source is explained in
[Current `random_state` constructor issue](#current-random_state-constructor-issue).

---

# `stacking_classifier`

`StackingClassifierCombiner` is registered as:

```text
stacking_classifier
```

It learns a second-level classifier from the F-Step prediction matrix.

Conceptually:

\[
P
\rightarrow
g_{\text{meta}}
\rightarrow
\widehat y.
\]

---

## Default meta-classifier

The constructor is:

```python
StackingClassifierCombiner(
    meta_model=None,
)
```

If no meta-model is provided, it creates:

```python
LogisticRegression(
    max_iter=1000,
)
```

The current implementation stores this configuration object as:

```python
self.meta_model
```

---

## Fit behavior

Before fitting, the source clones the configured meta-classifier:

```python
self.meta_model_ = clone(
    self.meta_model
)
```

Then it trains:

```python
self.meta_model_.fit(
    X,
    y,
)
```

and marks:

```python
self._is_fitted = True
```

---

## Why clone?

The configured estimator and fitted estimator are kept separate:

```text
meta_model
    constructor configuration

meta_model_
    fitted clone
```

This follows common scikit-learn fitted-state conventions.

---

## Prediction

`combine()` first checks:

```python
self._is_fitted
```

If the combiner has not been fitted, it raises:

```text
RuntimeError:
StackingClassifierCombiner is not fitted.
```

Otherwise it delegates to:

```python
self.meta_model_.predict(
    np.asarray(X)
)
```

---

## Direct example

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

## Custom meta-classifier

Any compatible scikit-learn-style classifier can be supplied.

```python
from sklearn.ensemble import RandomForestClassifier

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

After fitting:

```python
print(
    combiner.meta_model_
)
```

returns the fitted clone.

---

## KFC example

Use an instance with the current C-Step implementation:

```python
from kfc_procedure import KFCClassifier
from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner=StackingClassifierCombiner(),
    random_state=42,
)
```

---

## Inspect fitted stacking state

After fitting:

```python
strategy = model.cstep_.strategy_
```

Then:

```python
print(
    strategy.meta_model_
)
```

For the default logistic-regression meta-classifier:

```python
print(
    strategy.meta_model_.classes_
)

print(
    strategy.meta_model_.coef_
)
```

---

## No combiner-level `predict_proba()`

The current `StackingClassifierCombiner` implements:

```python
fit()
combine()
```

but does **not** implement:

```python
predict_proba()
```

So even if its underlying `meta_model_` supports probability prediction,
`CStep.predict_proba()` does not expose that capability through this combiner.

---

# `combined_classifier`

`CobraClassifierCombiner` is registered as:

```text
combined_classifier
```

It wraps the package's:

```python
CombinedClassifier
```

implementation.

Its constructor is:

```python
CobraClassifierCombiner(
    **cobra_params
)
```

and it creates:

```python
self.cobra = CombinedClassifier(
    **cobra_params
)
```

---

## KFC integration

The key fit call is:

```python
self.cobra.fit(
    X,
    y,
    as_predictions=True,
)
```

Here `X` is the F-Step prediction matrix.

That means the wrapped `CombinedClassifier` does not fit another internal set
of base classifiers when used as the KFC C-Step.

Instead, the vector of divergence-level class predictions is treated directly
as prediction-space data.

---

## Prediction

The wrapper delegates:

```python
return self.cobra.predict(X)
```

---

## Probability prediction

Unlike the other two current classification combiners,
`CobraClassifierCombiner` explicitly implements:

```python
predict_proba()
```

by calling:

```python
self.cobra.predict_proba(X)
```

So the C-Step strategy itself supports probability prediction when this
combiner is used.

---

## Example

```python
from kfc_procedure import KFCClassifier

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="combined_classifier",
    combiner_params={
        "distance": "hamming",
        "kernel": "rbf",
        "n_cv": 5,
    },
    random_state=42,
)
```

This string-based path is compatible with the current C-Step constructor logic
because `CobraClassifierCombiner` accepts arbitrary:

```python
**cobra_params
```

including the automatically forwarded `random_state`.

---

## Inspect the wrapped CombinedClassifier

After fitting:

```python
strategy = model.cstep_.strategy_

cobra = strategy.cobra

print(
    cobra
)
```

Depending on configuration and source version, useful fitted state can include
attributes such as:

```python
cobra.classes_
cobra.bandwidth_
cobra.pred_l_
cobra.distance_matrix_
cobra.global_majority_class_
cobra.optimization_outputs_
```

when those attributes are created by the wrapped estimator.

For the detailed algorithm, see:

[CombinedClassifier](../../getting-started/concepts/combined-classifier.md)

---

# Comparison

| Combiner | Learns from target? | Learned state | `predict_proba()` on combiner |
| --- | :---: | --- | :---: |
| `majority_vote` | No | none | No |
| `stacking_classifier` | Yes | `meta_model_` | No |
| `combined_classifier` | Yes | wrapped `CombinedClassifier` | Yes |

---

# Current `random_state` constructor issue

The current C-Step resolves a string combiner by copying:

```python
combiner_params
```

and then injecting:

```python
random_state=self.random_state
```

whenever the key is missing.

The relevant logic is:

```python
params = dict(
    self.combiner_params
)

if "random_state" not in params:
    params["random_state"] = self.random_state
```

Then:

```python
CombinerFactory.create(
    name,
    **params
)
```

is called.

---

## Affected classification combiners

These constructors do not accept `random_state`:

```python
MajorityVoteCombiner()
```

and:

```python
StackingClassifierCombiner(
    meta_model=None
)
```

Therefore a string-based configuration can result in calls equivalent to:

```python
MajorityVoteCombiner(
    random_state=42
)
```

or:

```python
StackingClassifierCombiner(
    random_state=42
)
```

and raise:

```text
TypeError:
... got an unexpected keyword argument 'random_state'
```

This can happen even when the forwarded value is `None`.

---

## `combined_classifier` is not affected

Its wrapper constructor is:

```python
def __init__(
    self,
    **cobra_params,
):
    ...
```

so C-Step's injected `random_state` is accepted and forwarded to
`CombinedClassifier`.

---

# Workaround for majority vote and stacking

Pass a combiner instance instead of its string name.

## Majority vote

```python
from kfc_procedure.core.combiner.classification import (
    MajorityVoteCombiner,
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner=MajorityVoteCombiner(),
    random_state=42,
)
```

---

## Stacking

```python
from kfc_procedure.core.combiner.classification import (
    StackingClassifierCombiner,
)

model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner=StackingClassifierCombiner(),
    random_state=42,
)
```

C-Step handles a non-string object with:

```python
if not isinstance(
    self.combiner,
    str,
):
    return self.combiner
```

so it does not reconstruct the object or inject constructor parameters.

---

# Direct C-Step use

You can use classification combiners directly with a prediction matrix.

For example:

```python
import numpy as np

P_train = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
])

y_train = np.array([
    0,
    1,
    0,
    1,
])
```

Then:

```python
combiner.fit(
    P_train,
    y_train,
)
```

and:

```python
prediction = combiner.predict(
    P_test
)
```

use the same interface employed by C-Step.

---

# Prediction matrix shape

Classification combiners expect rows to be samples and columns to be prediction
sources.

In KFC:

```text
(n_samples, n_divergences)
```

For example, with four divergences:

```text
(n_samples, 4)
```

The source explicitly validates two-dimensional input in:

```text
MajorityVoteCombiner
StackingClassifierCombiner.fit()
```

`CobraClassifierCombiner` delegates validation to the wrapped
`CombinedClassifier`.

---

# One divergence

A classifier may use only one divergence:

```python
divergences=[
    "euclidean"
]
```

Then the F-Step matrix has shape:

```text
(n_samples, 1)
```

With majority vote, that single prediction is simply the most frequent value
of a row containing one element, so the output matches that input label.

With stacking or CombinedClassifier, a learned mapping can still be fitted
from the one-column prediction representation.

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

the classification C-Step receives four labels per sample.

Example:

```text
[0, 0, 1, 0]
```

The selected combiner then decides how that prediction pattern becomes the
final class.

---

# C-Step task validation

For a string-based combiner, C-Step checks:

```python
CombinerFactory.supports(
    name,
    self.task,
)
```

So the following is valid:

```python
task="classification"
combiner="combined_classifier"
```

while a regression combiner such as:

```python
combiner="mean"
```

is rejected for classification.

---

# Inspect available classification combiners

```python
from kfc_procedure.core.combiner import (
    CombinerFactory,
)

print(
    CombinerFactory.available_by_category(
        "classification"
    )
)
```

The current source should include:

```text
combined_classifier
majority_vote
stacking_classifier
```

subject to normal registry loading.

---

# Inspect a fitted strategy

After KFC fitting:

```python
strategy = model.cstep_.strategy_

print(
    type(strategy).__name__
)
```

Possible values are:

```text
MajorityVoteCombiner
StackingClassifierCombiner
CobraClassifierCombiner
```

---

# Reconstruct classification consensus manually

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

manual = model.cstep_.predict(
    P
)

automatic = model.predict(
    X_test
)
```

Then:

```python
import numpy as np

print(
    np.array_equal(
        manual,
        automatic,
    )
)
```

should normally return:

```text
True
```

for the same fitted model and input.

---

# Inspect prediction patterns

A useful debugging step is to pair the F-Step pattern with the final output.

```python
divergence_names = list(
    model.fstep_.models_.keys()
)

for i in range(
    min(
        10,
        len(P),
    )
):
    print(
        dict(
            zip(
                divergence_names,
                P[i],
            )
        ),
        "->",
        manual[i],
    )
```

For majority vote, this shows the exact labels being counted.

For stacking or CombinedClassifier, it shows the feature vector seen by the
learned consensus strategy.

---

# Stacking with integer class predictions

The default stacking meta-model is logistic regression.

The F-Step matrix is therefore passed as a numeric feature matrix.

For example:

```text
[0, 0, 1, 0]
```

is treated as a vector of numeric predictor values.

The source does not one-hot encode divergence-level class labels before fitting
the meta-classifier.

!!! note

    This is an important implementation detail.

    The current stacking combiner treats the prediction matrix entries
    directly as numeric features.

---

# Arbitrary class-label caveat upstream

`MajorityVoteCombiner` itself allocates:

```python
dtype=object
```

for its outputs, so it can represent string or object labels.

However, the current **F-Step** initializes each divergence prediction vector
with:

```python
np.full(
    X.shape[0],
    np.nan,
)
```

which creates a floating-point array by default.

Therefore assigning string-valued local-classifier predictions into the
F-Step buffer can fail before the C-Step receives them.

!!! warning "Current pipeline limitation"

    Numeric class labels are the safest choice with the current full KFC
    classification pipeline.

    The combiner layer is more permissive than the current F-Step prediction
    buffer.

---

# Probability support

`CStep.predict_proba()` first checks:

```python
task == "classification"
```

and then checks whether the fitted strategy has:

```python
predict_proba
```

Among the current classification combiners:

```text
majority_vote       -> no
stacking_classifier -> no
combined_classifier -> yes
```

---

## End-to-end KFC probability limitation

Even though `combined_classifier` supports C-Step probabilities,
`KFCClassifier.predict_proba()` currently attempts:

```python
self.fstep_.predict_proba(
    X,
    clusters,
)
```

but the current `FStep` does not define:

```python
predict_proba()
```

Therefore:

```python
model.predict_proba(X)
```

is not currently functional end to end.

This limitation occurs before the C-Step's `combined_classifier`
`predict_proba()` can be used in the top-level KFC path.

---

# Calling CombinedClassifier probabilities directly

If you are working specifically with the fitted C-Step wrapper and already
have a suitable prediction matrix `P`, the strategy itself exposes:

```python
strategy.predict_proba(P)
```

for:

```python
CobraClassifierCombiner
```

Example:

```python
strategy = model.cstep_.strategy_

proba = strategy.predict_proba(
    P
)
```

This bypasses the missing F-Step probability method because `P` has already
been constructed.

---

# Learned combiners train on the aggregation subset

The KFC training flow separates the data internally:

```text
D_k
    -> K-Step
    -> F-Step

D_l
    -> fitted K-Step
    -> fitted F-Step
    -> prediction matrix P_l
    -> C-Step
```

So:

```text
stacking_classifier
combined_classifier
```

are trained on F-Step prediction patterns generated for the internal
aggregation subset.

`majority_vote` is stateless and does not use `y_l` during its own fitting.

---

# Majority vote vs learned consensus

The source provides different mechanisms rather than ranking them.

```text
majority_vote
    fixed row-wise mode

stacking_classifier
    learned meta-classifier

combined_classifier
    learned prediction-space consensus
```

Use whichever aggregation behavior you want to evaluate for the application.

---

# Custom classification combiner

A custom strategy should subclass:

```python
BaseCombiner
```

and implement:

```python
fit()
combine()
```

For a string-registered combiner, include `random_state` compatibility with the
current C-Step.

Example:

```python
import numpy as np
from collections import Counter

from kfc_procedure.core.combiner.base import (
    BaseCombiner,
)

from kfc_procedure.core.combiner import (
    CombinerFactory,
)


@CombinerFactory.register(
    "first_mode",
    categories={"classification"},
)
class FirstModeCombiner(
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

        output = np.empty(
            len(X),
            dtype=object,
        )

        for i, row in enumerate(X):
            output[i] = Counter(
                row
            ).most_common(1)[0][0]

        return output
```

---

# Custom probability combiner

A classification combiner can additionally implement:

```python
predict_proba()
```

For example:

```python
class MyProbabilityCombiner(
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
        y,
    ):
        ...
        return self

    def combine(
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

Then `CStep.predict_proba()` can delegate to it when the task is
classification.

The full KFC path still needs F-Step probability support before top-level
`KFCClassifier.predict_proba()` can work.

---

# Debugging classification consensus

## 1. Inspect the F-Step matrix

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)

print(
    P.shape
)

print(
    P[:10]
)
```

---

## 2. Check divergence column order

```python
print(
    list(
        model.fstep_.models_.keys()
    )
)
```

---

## 3. Inspect the strategy

```python
print(
    type(
        model.cstep_.strategy_
    ).__name__
)
```

---

## 4. Inspect stacking meta-model

```python
strategy = model.cstep_.strategy_

print(
    strategy.meta_model_
)
```

when using stacking.

---

## 5. Inspect CombinedClassifier

```python
strategy = model.cstep_.strategy_

print(
    strategy.cobra
)
```

when using `combined_classifier`.

---

## 6. Check string-constructor errors

If the error is:

```text
unexpected keyword argument 'random_state'
```

and the configured string is:

```text
majority_vote
stacking_classifier
```

pass a pre-instantiated combiner object with the current source version.

---

# Quick reference

| Name | Constructor | Stateless? | Probability support | Current string path |
| --- | --- | :---: | :---: | --- |
| `majority_vote` | `MajorityVoteCombiner()` | Yes | No | affected by `random_state` injection |
| `stacking_classifier` | `StackingClassifierCombiner(meta_model=None)` | No | No | affected |
| `combined_classifier` | `CobraClassifierCombiner(**cobra_params)` | No | Yes | compatible |

---

# Mental model

!!! quote ""

    **Classification consensus receives several divergence-specific class
    decisions and turns the resulting prediction pattern into one final class.**

\[
\boxed{
\begin{bmatrix}
c_1(x) &
c_2(x) &
\cdots &
c_M(x)
\end{bmatrix}
\rightarrow
g
\rightarrow
\widehat c(x)
}
\]

The current source provides a fixed hard vote, a learned stacking classifier,
and a `CombinedClassifier` prediction-space wrapper.

