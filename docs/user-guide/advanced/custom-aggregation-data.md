# Custom Aggregation Data

The COBRA estimators in `kfc_procedure` can use a **user-supplied aggregation
or calibration dataset** instead of automatically splitting one dataset into
training and calibration subsets.

The relevant fit arguments are:

```python
X_l=
y_l=
```

and they are supported by:

```text
GradientCOBRA
MixCOBRARegressor
CombinedClassifier
```

through the shared:

```python
resolve_training_context()
```

helper.

This advanced mode lets you decide exactly which observations are used for:

```text
base-estimator fitting
```

and which observations are used for:

```text
COBRA aggregation / calibration
```

---

## Core idea

In normal COBRA fitting, the package can split:

\[
(X,y)
\]

into:

\[
(X_k,y_k)
\]

for fitting base estimators and:

\[
(X_l,y_l)
\]

for constructing the prediction-space aggregation problem.

With explicit aggregation data, you provide those two roles yourself.

Conceptually:

```text
X_k, y_k
    ↓
fit base estimators
    ↓

X_l
    ↓
base-estimator predictions
    ↓
prediction-space calibration matrix
    ↓
distance / kernel / aggregation optimization
    ↑
y_l
```

---

# Explicit aggregation mode

The shared resolver uses explicit mode when both:

```python
X_l
```

and:

```python
y_l
```

are supplied.

The source branch is:

```python
if X_l is not None and y_l is not None:
    X_l, y_l = check_X_y(
        X_l,
        y_l,
    )

    return TrainingContext(
        X_k=np.asarray(X),
        y_k=np.asarray(y),
        X_l=np.asarray(X_l),
        y_l=np.asarray(y_l),
        as_predictions=False,
    )
```

Therefore the main `X` and `y` arguments become:

```text
X_k
y_k
```

while your explicit aggregation arrays become:

```text
X_l
y_l.
```

---

## Meaning of the four arrays

| Array | Current source role |
| --- | --- |
| `X` | base-estimator training features |
| `y` | base-estimator training targets |
| `X_l` | aggregation/calibration features |
| `y_l` | aggregation/calibration targets |

The resulting internal context is:

```text
X_k_ = X
y_k_ = y
X_l_ = X_l
y_l_ = y_l
as_predictions_ = False
```

for the ordinary feature-based path.

---

# No automatic first-level split

When explicit:

```python
X_l
y_l
```

are supplied, the resolver returns before the automatic splitter branch.

Therefore:

```text
split_ratio
overlap
```

do not generate another train/calibration split.

Your supplied arrays define the split directly.

---

# Both arrays are required

The resolver checks:

```python
if (
    X_l is None
) != (
    y_l is None
):
    raise ValueError(
        "Both 'X_l' and 'y_l' must be provided together."
    )
```

So these are invalid:

```python
model.fit(
    X,
    y,
    X_l=X_cal,
)
```

and:

```python
model.fit(
    X,
    y,
    y_l=y_cal,
)
```

Both calibration features and calibration targets must be supplied together.

---

# Input validation

The primary training data are first validated with:

```python
X, y = check_X_y(
    X,
    y,
)
```

The explicit aggregation data are separately validated with:

```python
X_l, y_l = check_X_y(
    X_l,
    y_l,
)
```

So each pair must have internally matching sample counts.

---

## Separate sample counts are allowed

The source does not require:

```python
len(X)
==
len(X_l)
```

because the two datasets serve different roles.

For example:

```text
X / y:
    1000 training rows

X_l / y_l:
    250 calibration rows
```

is structurally valid.

---

# Feature count must still be compatible

Although the resolver validates each pair separately, it does not explicitly
check that:

```python
X.shape[1]
==
X_l.shape[1]
```

before returning the context.

In ordinary feature mode, the fitted base estimators are trained on:

```python
X_k_
```

and later asked to predict:

```python
X_l_.
```

Therefore the feature dimensions must be compatible with those estimators.

A mismatch normally fails later through the underlying estimator's
`predict()` validation.

---

# Why custom aggregation data can be useful

The source supports explicit aggregation data without imposing a particular
reason for using it.

Examples include:

```text
a dedicated calibration dataset
a chronological holdout
a manually stratified split
a group-aware split prepared outside the package
a split produced by another library
a reproducible experiment with fixed train/calibration rows
```

The package accepts the supplied arrays and skips its own automatic
first-level splitter.

---

# GradientCOBRA

`GradientCOBRA.fit()` exposes:

```python
fit(
    X,
    y,
    X_l=None,
    y_l=None,
    split_ratio=0.5,
    overlap=0.0,
    as_predictions=False,
)
```

So explicit aggregation data can be supplied directly.

---

## GradientCOBRA example

```python
from kfc_procedure.cobra import (
    GradientCOBRA,
)

model = GradientCOBRA(
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

The internal context becomes:

```text
X_k_ = X_train
y_k_ = y_train

X_l_ = X_cal
y_l_ = y_cal
```

---

# GradientCOBRA training flow

With explicit aggregation data, the source does:

```python
self.estimators_ = self._fit_estimators(
    self.X_k_,
    self.y_k_,
)
```

then:

```python
prediction_space = self._load_predictions(
    self.X_l_,
)
```

So the base estimators are fitted on:

```text
X_train / y_train
```

and prediction-space coordinates are generated on:

```text
X_cal.
```

---

# GradientCOBRA calibration targets

The source sets:

```python
self.global_mean_ = float(
    np.mean(
        self.y_l_
    )
)
```

So the final zero-weight fallback is based on the explicit calibration targets:

```text
y_cal
```

not on the base-estimator training targets:

```text
y_train.
```

---

# GradientCOBRA cross-validation

After building the calibration prediction space, GradientCOBRA creates:

```python
self.cv_folds_ = list(
    self.cv_.split(
        self.X_l_,
        self.Y_l_norm_,
    )
)
```

These CV indices refer only to rows of:

```text
X_l_
```

and the normalized prediction-space matrix built from those calibration rows.

So the explicit aggregation dataset becomes the population used for COBRA
hyperparameter cross-validation.

---

# GradientCOBRA normalization detail

The current source computes:

```python
self.normalize_constant_ = (
    compute_normalization_constant(
        y=y,
        norm_constant=self.norm_constant,
        scale_factor=30.0,
        M=prediction_space.shape[1],
    )
)
```

Notice that it passes the original fit argument:

```python
y
```

rather than:

```python
self.y_l_.
```

---

## Consequence with explicit aggregation data

If you call:

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

then:

```text
prediction space
    is generated from X_cal

global_mean_
    is computed from y_cal

normalization constant
    is computed from y_train.
```

!!! important "Current source detail"

    GradientCOBRA's prediction-space normalization scalar is derived from the
    main `y` argument, even when explicit `y_l` calibration targets are
    supplied.

    This page documents the current implementation rather than assuming the
    normalization should instead use `y_l`.

---

# GradientCOBRA full example

```python
from sklearn.model_selection import (
    train_test_split,
)

from kfc_procedure.cobra import (
    GradientCOBRA,
)


X_train, X_cal, y_train, y_cal = (
    train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )
)


model = GradientCOBRA(
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)

y_pred = model.predict(
    X_test
)
```

The external `train_test_split()` defines the first-level separation.

GradientCOBRA does not split those arrays again before fitting base estimators
and constructing calibration prediction space.

---

# CombinedClassifier

`CombinedClassifier.fit()` exposes:

```python
fit(
    X,
    y,
    X_l=None,
    y_l=None,
    split_ratio=0.5,
    overlap=False,
    as_predictions=False,
)
```

Explicit calibration data follow the same resolver path.

---

## CombinedClassifier example

```python
from kfc_procedure.cobra import (
    CombinedClassifier,
)

classifier = CombinedClassifier(
    random_state=42,
)

classifier.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

---

# CombinedClassifier estimator fitting

In ordinary feature mode:

```python
self.classes_ = np.unique(
    self.y_k_
)

self.estimators_ = self._fit_estimators(
    self.X_k_,
    self.y_k_,
)

self.pred_l_ = self._load_predictions(
    self.X_l_
)
```

Therefore:

```text
base classifiers
    fit on X_train / y_train

prediction space
    generated on X_cal.
```

---

# CombinedClassifier classes

With explicit calibration data, `classes_` are derived from:

```python
self.y_k_
```

because:

```python
as_predictions_ == False.
```

So:

```python
self.classes_ = np.unique(
    y_train
)
```

rather than:

```python
np.unique(
    y_cal
)
```

---

# Global majority class

The fallback class is derived separately from:

```python
self.y_l_
```

using:

```python
classes, counts = np.unique(
    self.y_l_,
    return_counts=True,
)

self.global_majority_class_ = (
    classes[
        np.argmax(
            counts
        )
    ]
)
```

Therefore:

```text
classes_
    come from y_train

global_majority_class_
    comes from y_cal.
```

---

## Missing calibration classes

This distinction matters if:

```text
y_train
```

contains a class that is absent from:

```text
y_cal
```

or vice versa.

The source does not add a dedicated validation that both target arrays contain
exactly the same class set.

A user-supplied explicit classification split should therefore preserve the
class coverage needed by the experiment.

---

# CombinedClassifier cross-validation

The classifier creates folds with:

```python
self.cv_.split(
    self.X_l_,
    self.y_l_,
)
```

So bandwidth optimization occurs entirely within the explicit calibration
dataset.

The current high-level implementation still uses its ordinary internal
`KFoldCV`; supplying custom aggregation data does not change the CV strategy.

---

# CombinedClassifier default split is bypassed

When `X_l` and `y_l` are supplied, these arguments become irrelevant to the
first-level split:

```text
split_ratio
overlap.
```

The explicit arrays take precedence.

---

# Classification example with external stratification

The built-in COBRA automatic split is not stratified.

If you want a stratified first-level split, prepare it externally.

```python
from sklearn.model_selection import (
    train_test_split,
)

X_train, X_cal, y_train, y_cal = (
    train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
)
```

Then:

```python
classifier.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

This uses your stratified split instead of the package's automatic
`OverlapSplitter`.

---

# MixCOBRARegressor

`MixCOBRARegressor.fit()` exposes:

```python
fit(
    X,
    y,
    X_l=None,
    y_l=None,
    split_ratio=0.5,
    overlap=0.0,
    pred_features=None,
    as_predictions=False,
)
```

Explicit calibration data work through the same resolver.

---

## MixCOBRA example

```python
from kfc_procedure.cobra import (
    MixCOBRARegressor,
)

model = MixCOBRARegressor(
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

---

# MixCOBRA base estimators

With:

```python
as_predictions=False
```

the source runs:

```python
self.estimators_ = self._fit_estimators(
    self.X_k_,
    self.y_k_,
)
```

then:

```python
prediction_space = self._load_predictions(
    self.X_l_
)
```

So the role split is:

```text
X_train / y_train
    -> fit base regressors

X_cal
    -> generate prediction-space features.
```

---

# MixCOBRA input-space calibration data

MixCOBRA also retains:

```python
self.X_l_
```

because it compares both:

```text
input-space geometry
prediction-space geometry.
```

The explicit `X_cal` therefore serves two roles:

```text
raw calibration input space
source rows for base-model predictions.
```

---

# MixCOBRA normalization detail

The current source computes:

```python
self.normalize_constant_x_ = (
    compute_normalization_constant(
        X,
        norm_constant=self.norm_constant_x,
        scale_factor=5.0,
        M=prediction_space.shape[1],
    )
)
```

and:

```python
self.normalize_constant_y_ = (
    compute_normalization_constant(
        y,
        norm_constant=self.norm_constant_y,
        scale_factor=50.0,
        M=prediction_space.shape[1],
    )
)
```

---

## Consequence with explicit aggregation data

For:

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

the source computes:

```text
normalize_constant_x_
    from X_train

normalize_constant_y_
    from y_train
```

but applies them to:

```text
X_cal
prediction-space predictions on X_cal.
```

---

## Current source geometry

The stored arrays become:

```python
self.X_l_norm_ = (
    X_cal
    *
    normalize_constant_x_
)
```

and:

```python
self.Y_l_norm_ = (
    prediction_space_on_X_cal
    *
    normalize_constant_y_
)
```

This is the exact current implementation.

---

# MixCOBRA calibration target fallback

Like GradientCOBRA:

```python
self.global_mean_ = float(
    np.mean(
        self.y_l_
    )
)
```

So the final regression fallback comes from:

```text
y_cal
```

rather than:

```text
y_train.
```

---

# MixCOBRA cross-validation

The source creates folds over:

```python
self.X_l_
self.y_l_
```

so:

```text
alpha
beta
bandwidth-like parameters
```

are optimized on the explicit aggregation/calibration dataset.

---

# Automatic splitting vs explicit aggregation

Compare:

## Automatic

```python
model.fit(
    X,
    y,
    split_ratio=0.5,
    overlap=0.0,
)
```

The package chooses:

```text
X_k/y_k
X_l/y_l.
```

## Explicit

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

You choose those arrays.

The second form bypasses the package's automatic first-level split.

---

# Explicit aggregation vs precomputed predictions

These are different modes.

## Explicit aggregation features

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

means:

```text
fit internal estimators on X_train
predict X_cal internally
build prediction space from those predictions.
```

## Precomputed prediction mode

```python
model.fit(
    P_cal,
    y_cal,
    as_predictions=True,
)
```

means:

```text
do not fit internal estimators
treat P_cal itself as prediction space.
```

See:

[Precomputed Predictions](precomputed-predictions.md)

---

# Explicit aggregation and `as_predictions=True`

Because the resolver checks:

```python
if as_predictions:
    return ...
```

before it checks:

```python
X_l
y_l,
```

the precomputed branch takes precedence.

For example:

```python
model.fit(
    P,
    y,
    X_l=X_cal,
    y_l=y_cal,
    as_predictions=True,
)
```

causes the resolver to return:

```text
X_l = P
y_l = y
X_k = None
y_k = None
```

and the explicit:

```text
X_l=X_cal
y_l=y_cal
```

arguments are not used by the resolver.

!!! important "Branch precedence"

    Do not combine `as_predictions=True` with explicit `X_l/y_l` expecting
    both modes to be active.

    In the current resolver, precomputed-prediction mode returns first.

---

# Explicit aggregation and `pred_features`

The resolver checks explicit:

```python
X_l
y_l
```

before:

```python
pred_features.
```

So if both are provided while:

```python
as_predictions=False,
```

the explicit aggregation branch wins and:

```python
pred_features
```

is ignored by the resolver.

---

# Resolver precedence

The current order is:

```text
1. validate X/y

2. if as_predictions:
       prediction mode

3. validate X_l/y_l pair

4. if X_l and y_l:
       explicit aggregation mode

5. if pred_features:
       pred_features mode

6. otherwise:
       automatic split mode
```

This order is important when advanced arguments are combined.

---

# Explicit aggregation does not disable COBRA CV

Supplying:

```python
X_l
y_l
```

bypasses only the first-level automatic train/calibration split.

The estimator still performs:

```text
calibration-space construction
distance computation
cross-validation folds
hyperparameter optimization.
```

So the explicit calibration set is itself partitioned internally during COBRA
hyperparameter tuning.

---

# Use a dedicated calibration dataset

A common structure is:

```text
training set
    -> fit base estimators

calibration set
    -> fit COBRA aggregation geometry / tune aggregation parameters

test set
    -> final evaluation
```

The package supports the first two roles directly through:

```python
X, y, X_l, y_l
```

while the test set remains outside `fit()` and is passed later to:

```python
predict().
```

---

# Three-way split example

```python
from sklearn.model_selection import (
    train_test_split,
)

X_train_cal, X_test, y_train_cal, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
)

X_train, X_cal, y_train, y_cal = (
    train_test_split(
        X_train_cal,
        y_train_cal,
        test_size=0.25,
        random_state=42,
    )
)
```

This gives approximately:

```text
60% training
20% calibration
20% test.
```

Then:

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)

y_pred = model.predict(
    X_test
)
```

---

# Chronological aggregation data

For time-ordered data, you can prepare an earlier training block and a later
calibration block yourself.

```python
X_train = X[:train_end]
y_train = y[:train_end]

X_cal = X[
    train_end:cal_end
]

y_cal = y[
    train_end:cal_end
]
```

Then:

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
)
```

The COBRA estimator does not reshuffle this first-level split because explicit
aggregation data bypass the default splitter.

---

## Internal CV still matters

The current main COBRA estimators still build their normal internal:

```text
KFoldCV
```

over the calibration dataset.

Therefore using a chronological explicit aggregation set does **not** make the
internal hyperparameter CV time-series-aware.

The current high-level estimators do not expose a `cv=` argument.

See:

[Cross-validation](../cobra/cross-validation.md)

for that limitation.

---

# Group-aware external splitting

The current built-in splitter framework does not implement group-aware
partitioning.

You can prepare a group-aware split externally and pass the resulting arrays
as explicit aggregation data.

For example, use an external splitter to produce:

```text
train_idx
cal_idx
```

then:

```python
model.fit(
    X[train_idx],
    y[train_idx],
    X_l=X[cal_idx],
    y_l=y[cal_idx],
)
```

The estimator uses those rows directly.

---

# Class-balanced calibration data

For classification, explicit data can be useful when you need the first-level
calibration set to preserve class coverage.

The current `CombinedClassifier` automatic overlap splitter is not stratified.

Preparing:

```text
X_train/y_train
X_cal/y_cal
```

externally gives you control over that first-level class distribution.

---

# Class coverage checks

A practical source-aware diagnostic is:

```python
import numpy as np

print(
    np.unique(
        y_train,
        return_counts=True,
    )
)

print(
    np.unique(
        y_cal,
        return_counts=True,
    )
)
```

This matters because:

```text
classes_
```

and:

```text
global_majority_class_
```

are currently derived from different target arrays in feature-based
CombinedClassifier mode.

---

# Inspect resolved data after fitting

GradientCOBRA:

```python
print(
    model.X_k_.shape
)

print(
    model.X_l_.shape
)
```

MixCOBRA:

```python
print(
    model.X_k_.shape
)

print(
    model.X_l_.shape
)
```

CombinedClassifier:

```python
print(
    classifier.X_k_.shape
)

print(
    classifier.X_l_.shape
)
```

---

# Verify your explicit arrays were used

For NumPy arrays:

```python
import numpy as np

print(
    np.array_equal(
        model.X_k_,
        X_train,
    )
)

print(
    np.array_equal(
        model.X_l_,
        X_cal,
    )
)
```

The resolver converts values with:

```python
np.asarray()
```

so object identity is not the important check; compare values and shapes.

---

# Inspect calibration prediction space

GradientCOBRA:

```python
P_cal = model._load_predictions(
    model.X_l_
)

print(
    P_cal.shape
)
```

CombinedClassifier:

```python
print(
    classifier.pred_l_.shape
)
```

MixCOBRA:

```python
P_cal = model._load_predictions(
    model.X_l_
)

print(
    P_cal.shape
)
```

---

# Inspect calibration CV folds

```python
for fold in model.cv_folds_:
    print(
        fold.fold_id,
        len(
            fold.train_idx
        ),
        len(
            fold.eval_idx
        ),
    )
```

Those indices are relative to:

```text
X_l_
y_l_
```

not to the original full dataset.

---

# Mapping CV rows back to your calibration set

```python
fold = model.cv_folds_[0]

X_cal_train_fold = model.X_l_[
    fold.train_idx
]

X_cal_val_fold = model.X_l_[
    fold.eval_idx
]
```

This lets you inspect exactly which rows in your custom calibration set are
used for one internal optimization fold.

---

# Random state and explicit aggregation data

Supplying `X_l/y_l` removes randomness from the first-level COBRA split because
that split no longer occurs.

However, `random_state` can still affect:

```text
base estimators
internal KFoldCV shuffling
optimizer-related components
other estimator-specific randomness.
```

So an explicit aggregation set does not make the entire model deterministic by
itself.

---

# `split_ratio` is ignored for explicit data

This call:

```python
model.fit(
    X_train,
    y_train,
    X_l=X_cal,
    y_l=y_cal,
    split_ratio=0.9,
)
```

still uses your supplied:

```text
X_cal
y_cal.
```

The `split_ratio` value is never reached by the resolver's automatic-split
branch.

---

# `overlap` is ignored for explicit data

Likewise:

```python
overlap=0.2
```

does not cause rows to be copied between your explicit arrays.

If you want overlap between training and calibration data, you must construct
that overlap yourself before calling `fit()`.

---

# Intentional overlap

The source does not prevent the same observation from appearing in both:

```text
X
```

and:

```text
X_l.
```

For example:

```python
X_l = X_train[
    -100:
]

y_l = y_train[
    -100:
]
```

is accepted structurally.

The package does not track row identity and does not warn about reused
observations.

Any statistical consequences of that overlap are the caller's responsibility.

---

# No disjointness validation

The resolver validates shape consistency but does not test:

```text
duplicate rows
shared row IDs
sample provenance
train/calibration disjointness.
```

So "custom aggregation data" means the package trusts the split you provide.

---

# Calibration size

The source does not impose a package-specific minimum calibration sample count
before returning the explicit context.

However, later components can require enough rows for:

```text
n_cv folds
distance calculations
optimization
class representation.
```

For example, `n_cv=5` with a very small calibration set can produce empty or
degenerate folds in the current custom KFold implementation.

---

# Check calibration size before fitting

```python
print(
    len(
        X_cal
    )
)

print(
    model.n_cv
)
```

A calibration set should contain enough samples for the requested internal CV
structure.

The current source does not automatically reduce `n_cv` to match a small
calibration set.

---

# Regression fallback depends on calibration targets

Both GradientCOBRA and MixCOBRA compute:

```python
global_mean_ = np.mean(
    y_l_
)
```

So changing the custom aggregation dataset changes not only:

```text
distance geometry
CV objective
optimized parameters
```

but also the zero-weight fallback value.

---

# Classification fallback depends on calibration targets

CombinedClassifier computes:

```python
global_majority_class_
```

from:

```python
y_l_.
```

So a different calibration class distribution can change the fallback class
used when a query has no positive kernel-weight mass.

---

# Calibration data affect optimization

The optimized parameters are selected from cross-validation performed on the
calibration set.

Therefore changing:

```text
X_l
y_l
```

can change:

```text
bandwidth_
alpha / beta
optimization score
optimization history.
```

The source does not treat explicit aggregation data as a passive holdout; they
actively define the aggregation model.

---

# Calibration data are part of the fitted estimator

Attributes such as:

```text
X_l_
y_l_
prediction-space calibration arrays
distance matrices
CV folds
```

remain on the fitted estimator.

So the calibration data are not used only transiently during fit.

They define the reference set against which future query distances are
computed.

---

# Prediction compares against calibration representation

For GradientCOBRA:

```python
distance_matrix = self.distance_.matrix(
    Y_norm,
    self.Y_l_norm_,
)
```

For CombinedClassifier:

```python
distance_matrix = self.distance_.matrix(
    preds_space,
    self.pred_l_,
)
```

For MixCOBRA, query representations are compared with the stored normalized
calibration spaces.

Thus your explicit aggregation dataset becomes the model's prediction
reference population.

---

# Difference from a conventional validation set

A conventional validation set is often used only for model selection and then
discarded.

In the current COBRA implementation, the aggregation/calibration data also
serve as the stored reference observations used by the final prediction
mechanism.

So:

```text
X_l/y_l
```

are more than a temporary evaluation set.

They are part of the fitted aggregation model.

---

# Saving custom split indices

For reproducibility, it is often useful to save the row indices that produced:

```text
X_train/y_train
X_cal/y_cal.
```

For example:

```python
np.save(
    "train_idx.npy",
    train_idx,
)

np.save(
    "cal_idx.npy",
    cal_idx,
)
```

Then the same explicit aggregation configuration can be reconstructed later.

The package does not save original full-dataset indices automatically.

---

# DataFrames

If you pass pandas DataFrames and Series, the resolver uses:

```python
check_X_y()
```

and:

```python
np.asarray()
```

so the fitted internal context is NumPy-based.

Index labels are not retained as sample identifiers.

If row identity matters, keep the original DataFrame index or split indices
externally.

---

# Example: externally prepared holdout splitter

```python
from kfc_procedure.cobra.core.splitters import (
    RandomHoldoutSplitter,
)

splitter = RandomHoldoutSplitter(
    calibration_size=0.3,
    random_state=42,
)

indices = splitter.split(
    X,
    y,
)
```

Then:

```python
model.fit(
    X[
        indices.train_idx
    ],
    y[
        indices.train_idx
    ],
    X_l=X[
        indices.eval_idx
    ],
    y_l=y[
        indices.eval_idx
    ],
)
```

This is how you can use a splitter other than the high-level estimator's
hard-coded default first-level path.

---

# Example: custom index logic

```python
train_idx = np.arange(
    0,
    800,
)

cal_idx = np.arange(
    800,
    1000,
)

model.fit(
    X[
        train_idx
    ],
    y[
        train_idx
    ],
    X_l=X[
        cal_idx
    ],
    y_l=y[
        cal_idx
    ],
)
```

The package accepts the resulting arrays without needing to know how those
indices were selected.

---

# Custom aggregation data with KFC Procedure

The top-level KFC Procedure does not currently expose:

```python
X_l=
y_l=
```

through its public:

```python
fit()
```

interface.

KFC performs its own internal split, then its C-Step receives the internal
aggregation prediction matrix.

Therefore the custom aggregation-data interface documented here applies
directly to the standalone COBRA estimators, not to the top-level
`KFCProcedure.fit()` API.

---

# KFC C-Step wrappers

Inside KFC, wrappers such as:

```text
GradientCOBRACombiner
MixCOBRACombiner
CobraClassifierCombiner
```

receive the already-created F-Step matrix and call the wrapped estimator with:

```python
as_predictions=True.
```

That is a different advanced pathway from supplying raw feature-based
`X_l/y_l` to standalone COBRA.

---

# Custom aggregation vs KFC internal split

Standalone COBRA:

```text
you can provide X_l / y_l directly.
```

Top-level KFC:

```text
KFCProcedure.fit() creates its own internal K/F vs C split.
```

If you need full control over the KFC internal split, the current public KFC
fit interface does not expose that as `X_l/y_l`.

---

# Common mistake: using test data as aggregation data unintentionally

Because `X_l/y_l` become part of the fitted model and influence
hyperparameter optimization, they are not untouched final-evaluation data.

If you need an independent test set, keep it separate and call:

```python
predict(
    X_test
)
```

only after fitting.

---

# Common mistake: mismatched preprocessing

The base estimators are fitted on:

```text
X
```

and asked to predict:

```text
X_l.
```

If those arrays were produced with inconsistent external preprocessing, the
package does not reconcile them.

Ensure they use the same feature schema and transformation conventions.

---

# Common mistake: different column order

Even if:

```python
X.shape[1]
==
X_l.shape[1]
```

the semantics can still differ if columns were reordered.

The package does not store feature names after conversion to NumPy.

---

# Common mistake: too-small classification calibration set

A small:

```text
y_l
```

can omit classes or create unstable internal K-fold composition.

The source does not automatically repair class coverage.

Inspect:

```python
np.unique(
    y_l,
    return_counts=True,
)
```

before fitting.

---

# Common mistake: expecting `split_ratio` to modify explicit data

Once:

```python
X_l
y_l
```

are present, the resolver does not use:

```text
split_ratio
overlap.
```

Changing those parameters has no effect on your custom first-level split.

---

# Common mistake: combining `X_l/y_l` with `as_predictions=True`

Because `as_predictions=True` is checked first, your explicit calibration
arrays are bypassed.

Choose one mode deliberately:

```text
raw feature training + explicit aggregation features
```

or:

```text
already-precomputed prediction space.
```

---

# Debugging explicit aggregation mode

## Check mode

```python
print(
    model.as_predictions_
)
```

Expected:

```text
False
```

for raw-feature explicit `X_l/y_l` mode.

---

## Check stored arrays

```python
print(
    model.X_k_.shape
)

print(
    model.X_l_.shape
)
```

---

## Check calibration target summary

Regression:

```python
print(
    np.mean(
        model.y_l_
    )
)
```

Classification:

```python
print(
    np.unique(
        model.y_l_,
        return_counts=True,
    )
)
```

---

## Check internal estimator count

```python
print(
    len(
        model.estimators_
    )
)
```

In feature-based explicit aggregation mode, internal base estimators are still
fitted.

---

## Check calibration prediction-space shape

GradientCOBRA or MixCOBRA:

```python
P_l = model._load_predictions(
    model.X_l_
)

print(
    P_l.shape
)
```

CombinedClassifier:

```python
print(
    model.pred_l_.shape
)
```

---

## Check CV uses calibration rows

```python
for fold in model.cv_folds_:
    assert np.max(
        fold.eval_idx
    ) < len(
        model.X_l_
    )
```

for non-empty validation folds.

---

# Validation checklist

Before calling `fit()`:

```python
assert len(
    X_train
) == len(
    y_train
)

assert len(
    X_cal
) == len(
    y_cal
)
```

For ordinary feature mode:

```python
assert (
    X_train.shape[1]
    ==
    X_cal.shape[1]
)
```

For numeric regression:

```python
assert np.isfinite(
    X_train
).all()

assert np.isfinite(
    X_cal
).all()

assert np.isfinite(
    y_train
).all()

assert np.isfinite(
    y_cal
).all()
```

These checks are useful because the package's component-level validation is
not uniform.

---

# Current source comparison

| Estimator | Explicit `X_l/y_l` | Base estimators fit on | Aggregation representation built on |
| --- | :---: | --- | --- |
| `GradientCOBRA` | Yes | `X/y` | predictions on `X_l` |
| `MixCOBRARegressor` | Yes | `X/y` | `X_l` + predictions on `X_l` |
| `CombinedClassifier` | Yes | `X/y` | hard predictions on `X_l` |

---

# Calibration-derived fitted state

| Estimator | Calibration-derived state |
| --- | --- |
| `GradientCOBRA` | `X_l_`, `y_l_`, `Y_l_norm_`, `distance_matrix_`, `cv_folds_`, `global_mean_` |
| `MixCOBRARegressor` | `X_l_`, `y_l_`, normalized X/Y spaces, distance matrices, `cv_folds_`, `global_mean_` |
| `CombinedClassifier` | `X_l_`, `y_l_`, `pred_l_`, `distance_matrix_`, `cv_folds_`, `global_majority_class_` |

---

# Branch precedence quick reference

| Inputs | Resolver mode |
| --- | --- |
| `as_predictions=True` | precomputed prediction mode |
| `X_l` + `y_l` | explicit aggregation mode |
| `pred_features` only | `pred_features` mode |
| none of the above | automatic split mode |
| only one of `X_l` / `y_l` | error |

---

# Quick reference

| Goal | Current source usage |
| --- | --- |
| supply dedicated calibration features | `X_l=X_cal, y_l=y_cal` |
| bypass automatic overlap split | provide both `X_l` and `y_l` |
| keep internal base estimators | use explicit `X_l/y_l` with `as_predictions=False` |
| skip internal base estimators | use `as_predictions=True` instead |
| use external stratified first-level split | prepare arrays externally, then pass `X_l/y_l` |
| use external chronological first-level split | prepare arrays externally, then pass `X_l/y_l` |
| inspect resolved training rows | `X_k_`, `y_k_` |
| inspect resolved calibration rows | `X_l_`, `y_l_` |
| inspect internal calibration CV | `cv_folds_` |

---

# Mental model

!!! quote ""

    **Custom aggregation data lets you decide which observations train the
    model experts and which observations become the reference population for
    COBRA aggregation.**

\[
\boxed{
(X_k,y_k)
\rightarrow
\text{fit experts}
}
\]

\[
\boxed{
X_l
\rightarrow
\text{expert predictions}
\rightarrow
\text{COBRA geometry}
\rightarrow
\text{aggregation}
}
\]

with:

\[
\boxed{
y_l
\rightarrow
\text{CV objective + final reference targets}
}
\]

The explicit `X_l/y_l` path bypasses the automatic first-level split, but it
does not bypass the COBRA distance, cross-validation, optimization, or final
reference-set behavior.
