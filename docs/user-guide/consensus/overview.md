# Consensus Overview

The **consensus layer** is the final stage of the KFC Procedure.

After the K-Step creates divergence-specific partitions and the F-Step turns
those partitions into local predictions, the C-Step combines those predictions
into one final output.

\[
\boxed{
\text{K-Step}
\rightarrow
\text{F-Step}
\rightarrow
\text{C-Step consensus}
}
\]

The consensus implementation is built around:

```python
BaseCombiner
CombinerFactory
CStep
```

and the concrete strategies under:

```text
kfc_procedure/core/combiner/
├── regression/
└── classification/
```

---

## What consensus receives

The C-Step receives a prediction matrix:

\[
P
\in
\mathbb{R}^{n\times M},
\]

where:

- \(n\) is the number of observations;
- \(M\) is the number of divergence-specific prediction columns.

For regression:

```text
             euclidean      gkl      logistic
sample 1        12.4        12.8       12.1
sample 2        20.7        21.0       20.4
sample 3         8.3         8.1        8.5
```

For classification:

```text
             euclidean      gkl      logistic
sample 1          0           0           1
sample 2          1           1           1
sample 3          2           1           2
```

The consensus strategy then computes:

\[
P
\longrightarrow
\widehat y.
\]

The C-Step does not directly use the original feature matrix or K-Step
centroids. Its input is the F-Step prediction matrix.

---

## Consensus families

The current source registers five regression combiners and three
classification combiners.

### Regression

| Registry name | Class | Type |
| --- | --- | --- |
| `mean` | `MeanCombiner` | deterministic |
| `weighted_mean` | `WeightedMeanCombiner` | learned linear |
| `stacking_regressor` | `StackingRegressorCombiner` | meta-regression |
| `gradientcobra` | `GradientCOBRACombiner` | COBRA-style nonparametric |
| `mixcobra` | `MixCOBRACombiner` | COBRA-style nonparametric |

### Classification

| Registry name | Class | Type |
| --- | --- | --- |
| `majority_vote` | `MajorityVoteCombiner` | deterministic |
| `stacking_classifier` | `StackingClassifierCombiner` | meta-classification |
| `combined_classifier` | `CobraClassifierCombiner` | prediction-space consensus |

---

## Stateless vs learned consensus

<div class="grid cards" markdown>

-   :material-lightning-bolt-outline:{ .lg .middle } **Stateless**

    ---

    No model parameters are learned from the C-Step target values.

    ```text
    mean
    majority_vote
    ```

-   :material-school-outline:{ .lg .middle } **Learned**

    ---

    These strategies fit coefficients, a meta-estimator, or a COBRA model.

    ```text
    weighted_mean
    stacking_regressor
    stacking_classifier
    gradientcobra
    mixcobra
    combined_classifier
    ```

</div>

Even stateless strategies still receive a `fit()` call because all combiners
share the same interface.

---

# Regression consensus

Regression combiners operate on numeric F-Step predictions and return one
numeric prediction per row.

---

## Mean

`mean` computes the row-wise arithmetic mean:

\[
\widehat y_i
=
\frac{1}{M}
\sum_{m=1}^{M}P_{im}.
\]

The implementation is simply:

```python
np.mean(
    X,
    axis=1,
)
```

Example:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="mean",
)
```

No coefficients are learned.

See:

[Mean and Regression Combiners](regression-combiners.md)

---

## Weighted mean

`weighted_mean` uses scikit-learn `LinearRegression`.

The model is:

\[
y
\approx
Pw.
\]

By default:

```python
fit_intercept=False
```

so the fitted prediction is based only on the learned coefficients.

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
        "logistic",
    ],
    local_model="ridge",
    combiner="weighted_mean",
    combiner_params={
        "fit_intercept": False,
    },
)
```

!!! note

    The current implementation does not constrain the learned coefficients to
    be positive or to sum to one.

    `weighted_mean` is therefore an ordinary linear-regression combiner rather
    than a convex weighted average.

See:

[Weighted Mean](weighted-mean.md)

---

## Stacking regressor

`stacking_regressor` trains a meta-regressor on the F-Step prediction matrix.

Its default meta-model is:

```python
LinearRegression()
```

The source clones the configured meta-model before fitting:

```python
self.meta_model_ = clone(
    self.meta_model
)
```

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="stacking_regressor",
)
```

A custom meta-regressor can be supplied through `combiner_params`.

See:

[Stacking](stacking.md)

---

## GradientCOBRA

`gradientcobra` wraps:

```python
GradientCOBRA
```

The wrapper fits it with:

```python
self.cobra.fit(
    X,
    y,
    as_predictions=True,
)
```

so the KFC F-Step matrix is used directly as precomputed prediction-space
features.

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="gradientcobra",
    combiner_params={
        "distance": "euclidean",
        "kernel": "rbf",
    },
)
```

For the full algorithm, see:

[GradientCOBRA](../../getting-started/concepts/gradientcobra.md)

---

## MixCOBRA

`mixcobra` wraps:

```python
MixCOBRARegressor
```

and also fits it using:

```python
as_predictions=True
```

inside the KFC combiner.

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="mixcobra",
    combiner_params={
        "distance": "euclidean",
        "kernel": "rbf",
    },
)
```

!!! note

    In standalone MixCOBRA, the algorithm is designed around input-space and
    prediction-space information.

    In the KFC C-Step wrapper, the F-Step matrix is passed as precomputed
    predictions through `as_predictions=True`.

For the full algorithm, see:

[MixCOBRA](../../getting-started/concepts/mixcobra.md)

---

# Classification consensus

Classification combiners receive one predicted label per divergence and return
one final class label.

---

## Majority vote

`majority_vote` selects the most frequent label in each prediction row.

For:

```text
[1, 1, 0, 1]
```

the output is:

```text
1
```

The implementation uses:

```python
Counter(
    row
).most_common(1)[0][0]
```

Example:

```python
from kfc_procedure import KFCClassifier

model = KFCClassifier(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="logistic_regression",
    combiner="majority_vote",
)
```

The source does not implement a separate tie-resolution strategy.

See:

[Majority Vote](majority-vote.md)

---

## Stacking classifier

`stacking_classifier` trains a meta-classifier over the F-Step class-prediction
matrix.

The default meta-model is:

```python
LogisticRegression(
    max_iter=1000,
)
```

The configured meta-model is cloned before fitting.

Example:

```python
model = KFCClassifier(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="logistic_regression",
    combiner="stacking_classifier",
)
```

See:

[Stacking](stacking.md)

---

## CombinedClassifier

`combined_classifier` wraps:

```python
CombinedClassifier
```

and fits it on the F-Step matrix using:

```python
as_predictions=True
```

This makes the vector of divergence-specific class predictions the
prediction-space representation used for consensus.

Example:

```python
model = KFCClassifier(
    divergences=[
        "euclidean",
        "gkl",
        "logistic",
    ],
    local_model="logistic_regression",
    combiner="combined_classifier",
    combiner_params={
        "distance": "hamming",
        "kernel": "rbf",
    },
)
```

The wrapper also exposes:

```python
predict_proba()
```

by delegating to the underlying `CombinedClassifier`.

For the algorithm itself, see:

[CombinedClassifier](../../getting-started/concepts/combined-classifier.md)

---

# How C-Step resolves a combiner

The C-Step accepts:

```python
combiner
```

as either:

```text
a registered string
```

or:

```text
a pre-instantiated combiner object
```

For a string, it:

1. lowercases the name;
2. checks that the combiner exists;
3. checks task compatibility;
4. copies `combiner_params`;
5. injects `random_state` if missing;
6. creates the strategy through `CombinerFactory`.

---

## Task-aware validation

For regression:

```python
CombinerFactory.supports(
    name,
    "regression",
)
```

must be true.

For classification:

```python
CombinerFactory.supports(
    name,
    "classification",
)
```

must be true.

Therefore:

```python
combiner="mean"
```

is invalid for a classifier, and:

```python
combiner="majority_vote"
```

is invalid for a regressor.

---

# Current `random_state` constructor mismatch

The current C-Step unconditionally injects:

```python
random_state=self.random_state
```

for string-based combiners when the key is not already in `combiner_params`.

This affects several simple built-in combiners whose constructors do not
accept that keyword.

The affected constructors are:

```text
MeanCombiner
WeightedMeanCombiner
StackingRegressorCombiner
MajorityVoteCombiner
StackingClassifierCombiner
```

The COBRA wrappers are not affected because they accept arbitrary:

```python
**cobra_params
```

!!! warning "Current source behavior"

    In the current source version, the simple string-based combiners above can
    raise an `unexpected keyword argument 'random_state'` error when C-Step
    constructs them.

    This is an implementation issue in parameter forwarding, not a limitation
    of the consensus strategies themselves.

---

## Workaround with a combiner instance

Because C-Step returns a non-string combiner object directly, a
pre-instantiated strategy avoids the automatic keyword injection.

Regression:

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

Classification:

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

# Combiner API

Every built-in combiner inherits:

```python
BaseCombiner
```

which defines:

```python
fit(X, y=None)
combine(X)
predict(X)
```

The default `predict()` implementation simply calls:

```python
self.combine(X)
```

This keeps rule-based and learned aggregation strategies behind one common API.

---

# Input shape

The combiner contract expects:

```text
(n_samples, n_models)
```

or, in KFC terminology:

```text
(n_samples, n_divergences)
```

Several combiners explicitly reject one-dimensional input.

For example:

```python
if X.ndim != 2:
    raise ValueError(...)
```

is present in:

```text
MeanCombiner
WeightedMeanCombiner.fit()
StackingRegressorCombiner.fit()
MajorityVoteCombiner
StackingClassifierCombiner.fit()
```

---

# Prediction matrix column meaning

In KFC, each column corresponds to a divergence-specific F-Step prediction
family.

For example:

```text
column 0 -> euclidean
column 1 -> gkl
column 2 -> logistic
column 3 -> is
```

You can inspect the effective column order with:

```python
list(
    model.fstep_.models_.keys()
)
```

and reconstruct the C-Step input with:

```python
clusters = model.kstep_.predict(
    X_test
)

P = model.fstep_.predict(
    X_test,
    clusters,
)
```

---

# Inspect the fitted strategy

After fitting:

```python
strategy = model.cstep_.strategy_
```

Inspect its class:

```python
print(
    type(strategy).__name__
)
```

Examples:

```text
MeanCombiner
WeightedMeanCombiner
StackingRegressorCombiner
GradientCOBRACombiner
MixCOBRACombiner
MajorityVoteCombiner
StackingClassifierCombiner
CobraClassifierCombiner
```

---

# Inspect learned consensus state

Different strategies expose different fitted attributes.

| Strategy | Useful fitted state |
| --- | --- |
| `mean` | none |
| `weighted_mean` | `strategy.model.coef_` |
| `stacking_regressor` | `strategy.meta_model_` |
| `gradientcobra` | `strategy.cobra` |
| `mixcobra` | `strategy.cobra` |
| `majority_vote` | none |
| `stacking_classifier` | `strategy.meta_model_` |
| `combined_classifier` | `strategy.cobra` |

---

# Probability prediction

`CStep.predict_proba()` is available only when:

```python
task == "classification"
```

and the selected strategy itself exposes:

```python
predict_proba()
```

Among the current built-in classification combiners:

| Combiner | C-Step probability support |
| --- | :---: |
| `majority_vote` | No |
| `stacking_classifier` | No |
| `combined_classifier` | Yes |

---

## Top-level KFC probability limitation

Although `CobraClassifierCombiner` implements `predict_proba()`, the current
top-level KFC classification pipeline is incomplete for probability prediction.

`KFCClassifier.predict_proba()` attempts to call:

```python
self.fstep_.predict_proba(...)
```

but the current `FStep` has no such method.

Therefore end-to-end KFC probabilities are not currently functional in this
source version.

---

# Rule-based vs learned consensus

A useful way to understand the available strategies is:

```text
prediction matrix
       │
       ├── fixed rule
       │     ├── mean
       │     └── majority_vote
       │
       └── fitted rule
             ├── weighted_mean
             ├── stacking_regressor
             ├── stacking_classifier
             ├── gradientcobra
             ├── mixcobra
             └── combined_classifier
```

The source does not prescribe one strategy as universally preferable.

They provide different aggregation mechanisms over the same F-Step output.

---

# Regression strategy summary

| Strategy | Learns from `y`? | Aggregation mechanism |
| --- | :---: | --- |
| `mean` | No | arithmetic mean |
| `weighted_mean` | Yes | linear regression |
| `stacking_regressor` | Yes | configurable meta-regressor |
| `gradientcobra` | Yes | GradientCOBRA |
| `mixcobra` | Yes | MixCOBRARegressor |

See:

[Regression Combiners](regression-combiners.md)

---

# Classification strategy summary

| Strategy | Learns from `y`? | Aggregation mechanism |
| --- | :---: | --- |
| `majority_vote` | No | row-wise mode |
| `stacking_classifier` | Yes | configurable meta-classifier |
| `combined_classifier` | Yes | CombinedClassifier prediction-space aggregation |

See:

[Classification Combiners](classification-combiners.md)

---

# Consensus and the KFC split

KFC trains the clustering/local-model stages on one internal subset and trains
the C-Step on another.

Conceptually:

```text
Dₖ
│
├── K-Step
└── F-Step
       │
       ▼
fitted local prediction system

Dₗ
│
└── passed through K-Step + F-Step
       │
       ▼
      Pₗ
       │
       ▼
    C-Step
```

This means learned consensus methods fit on prediction vectors generated for
the internal aggregation subset rather than directly on the same rows used to
fit the cluster-local estimators.

---

# Custom consensus strategy

You can implement a custom combiner by subclassing:

```python
BaseCombiner
```

and implementing:

```python
fit()
combine()
```

Example:

```python
import numpy as np

from kfc_procedure.core.combiner.base import (
    BaseCombiner,
)


class MedianCombiner(BaseCombiner):

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

Then pass the instance directly:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner=MedianCombiner(),
)
```

---

# Register a custom combiner

A named custom combiner can be registered with:

```python
CombinerFactory
```

Example:

```python
from kfc_procedure.core.combiner import (
    CombinerFactory,
)


@CombinerFactory.register(
    "median",
    categories={"regression"},
)
class MedianCombiner(BaseCombiner):
    ...
```

Then:

```python
combiner="median"
```

can be resolved by C-Step.

!!! warning

    Because the current C-Step injects `random_state` into every string-based
    combiner, a custom registered combiner should accept:

    ```python
    random_state=None
    ```

    or arbitrary keyword arguments.

---

# Choosing a consensus family

A source-aligned way to think about the options is:

<div class="grid cards" markdown>

-   :material-equal:{ .lg .middle } **Fixed averaging / voting**

    ---

    Use a deterministic rule over prediction columns.

    ```text
    mean
    majority_vote
    ```

-   :material-weight:{ .lg .middle } **Linear weighting**

    ---

    Learn a linear mapping from the prediction matrix.

    ```text
    weighted_mean
    ```

-   :material-layers-triple:{ .lg .middle } **Stacking**

    ---

    Fit a second-level predictive model.

    ```text
    stacking_regressor
    stacking_classifier
    ```

-   :material-chart-bell-curve:{ .lg .middle } **COBRA-style consensus**

    ---

    Use prediction-space similarity and kernel aggregation.

    ```text
    gradientcobra
    mixcobra
    combined_classifier
    ```

</div>

---

# Debugging consensus

## Inspect the F-Step matrix

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
    P[:5]
)
```

---

## Inspect the combiner type

```python
print(
    type(
        model.cstep_.strategy_
    ).__name__
)
```

---

## Inspect factory choices

```python
from kfc_procedure.core.combiner import (
    CombinerFactory,
)

print(
    CombinerFactory.available_by_category(
        "regression"
    )
)

print(
    CombinerFactory.available_by_category(
        "classification"
    )
)
```

---

## Reconstruct final predictions

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

The two paths should represent the same fitted KFC prediction pipeline.

---

# Quick reference

| Task | Combiner | Registry |
| --- | --- | --- |
| Regression | arithmetic mean | `mean` |
| Regression | OLS weighting | `weighted_mean` |
| Regression | meta-regression | `stacking_regressor` |
| Regression | GradientCOBRA | `gradientcobra` |
| Regression | MixCOBRA | `mixcobra` |
| Classification | hard vote | `majority_vote` |
| Classification | meta-classification | `stacking_classifier` |
| Classification | CombinedClassifier | `combined_classifier` |

---

# Mental model

!!! quote ""

    **Consensus is the stage that turns several divergence-specific answers
    into one final answer.**

\[
\boxed{
\begin{bmatrix}
p_1(x) &
p_2(x) &
\cdots &
p_M(x)
\end{bmatrix}
\rightarrow
\text{consensus strategy}
\rightarrow
\widehat y(x)
}
\]

The strategy can be a fixed rule, a learned meta-model, or a COBRA-style
prediction-space aggregation method.

