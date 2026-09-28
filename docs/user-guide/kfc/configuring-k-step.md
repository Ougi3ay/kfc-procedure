# Configure K-Step

The **K-Step** is the clustering stage of the KFC Procedure.

Its job is to construct several candidate partitions of the same input data by
running one `BregmanKMeans` model for every configured Bregman divergence.

\[
\boxed{
X
\rightarrow
\text{Bregman K-means under several divergences}
\rightarrow
\text{multiple cluster assignments}
}
\]

The implementation is provided by:

```python
from kfc_procedure.core.steps import KStep
```

Most users configure it indirectly through:

```python
KFCRegressor(...)
KFCClassifier(...)
```

using the arguments:

```text
divergences
divergences_params
n_clusters
max_iter
tol
random_state
```

---

## Basic configuration

A minimal K-Step configuration uses one divergence:

```python
from kfc_procedure import KFCRegressor

model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mean",
    n_clusters=3,
    random_state=42,
)
```

During fitting, KFC creates a `KStep` equivalent to:

```python
KStep(
    divergences=["euclidean"],
    divergences_params={},
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=42,
)
```

The K-Step then fits one `BregmanKMeans` instance.

---

## Why several divergences?

Ordinary K-means assumes squared-Euclidean geometry.

KFC does not commit to a single geometry. Instead, it can fit several
clusterings independently:

```text
                         ┌─ Euclidean ─────► partition 1
                         │
Input X ─────────────────┼─ GKL ───────────► partition 2
                         │
                         ├─ Logistic ──────► partition 3
                         │
                         └─ Itakura-Saito ─► partition 4
```

Each partition is passed independently to the F-Step.

The K-Step itself does **not** attempt to reconcile or vote between cluster
assignments.

---

## Mathematical view

For divergences

\[
d_1,d_2,\ldots,d_M,
\]

the K-Step fits

\[
\mathcal{K}_m
=
\operatorname{BregmanKMeans}(d_m,K),
\qquad
m=1,\ldots,M,
\]

where \(K\) is the configured number of clusters.

Each model produces its own cluster assignment

\[
C_m(X)
=
\left(
c_{m,1},
\ldots,
c_{m,n}
\right).
\]

Therefore the same observation can belong to different clusters under different
divergences.

---

## Built-in divergences

The package currently registers four divergences.

| Name | Class | Exponential-family interpretation | Domain |
| --- | --- | --- | --- |
| `euclidean` | `SquaredEuclidean` | Gaussian | \(\mathbb{R}^d\) |
| `gkl` | `GKLDivergence` | Poisson | \((0,\infty)^d\) |
| `logistic` | `LogisticLoss` | Bernoulli / Binomial | \((0,1)^d\) |
| `is` | `ItakuraSaito` | Exponential / Gamma | \((0,\infty)^d\) |

Importing the divergence package registers these names with
`BregmanDivergenceFactory`.

---

## Squared Euclidean

Use:

```python
divergences=["euclidean"]
```

The generator is

\[
\phi(x)=\|x\|_2^2
\]

and the divergence is

\[
D(x,y)=\|x-y\|_2^2.
\]

Its domain is unrestricted:

\[
x\in\mathbb{R}^d.
\]

This is the safest starting point for arbitrary numerical features.

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mean",
    random_state=42,
)
```

---

## Generalized KL

Use:

```python
divergences=["gkl"]
```

The implementation uses

\[
D(x,y)
=
\sum_j
\left[
x_j\log\left(\frac{x_j}{y_j}\right)
-
(x_j-y_j)
\right].
\]

The required domain is

\[
x_j>0.
\]

Example:

```python
model = KFCRegressor(
    divergences=["gkl"],
    local_model="ridge",
    combiner="mean",
)
```

!!! warning

    Zero and negative feature values are rejected by the domain check.

---

## Logistic divergence

Use:

```python
divergences=["logistic"]
```

The implementation corresponds to the Bernoulli/Binomial Bregman divergence:

\[
D(x,y)
=
\sum_j
\left[
x_j\log\frac{x_j}{y_j}
+
(1-x_j)
\log\frac{1-x_j}{1-y_j}
\right].
\]

Its domain is strictly

\[
0<x_j<1.
\]

Example:

```python
model = KFCClassifier(
    divergences=["logistic"],
    local_model="logistic_regression",
    combiner="majority_vote",
)
```

!!! warning

    Values equal to `0` or `1` are outside the domain.

---

## Itakura–Saito

Use:

```python
divergences=["is"]
```

The implementation uses

\[
D(x,y)
=
\sum_j
\left[
\frac{x_j}{y_j}
-
\log\left(\frac{x_j}{y_j}\right)
-
1
\right].
\]

Its required domain is

\[
x_j>0.
\]

Example:

```python
model = KFCRegressor(
    divergences=["is"],
    local_model="ridge",
    combiner="mean",
)
```

---

## Using all divergences together

A common KFC configuration is:

```python
divergences=[
    "euclidean",
    "gkl",
    "logistic",
    "is",
]
```

Because every divergence receives the same input matrix, the data must satisfy
the intersection of all selected domains.

For all four built-ins, that means:

\[
0<x_{ij}<1.
\]

A practical preprocessing strategy is:

```python
from sklearn.preprocessing import MinMaxScaler

eps = 1e-6

scaler = MinMaxScaler(
    feature_range=(eps, 1.0 - eps),
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Then:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
        "logistic",
        "is",
    ],
    local_model="ridge",
    combiner="weighted_mean",
    n_clusters=3,
    random_state=42,
)

model.fit(X_train, y_train)
```

---

## Domain validation

`BregmanKMeans.fit()` validates the training data before clustering.

It rejects:

- `NaN`,
- infinite values,
- values outside the selected divergence's domain.

The validation logic raises errors such as:

```text
[GKL] Input outside valid domain.
```

or:

```text
[Logit] Input contains NaN or Inf.
```

Prediction also evaluates the divergence against fitted centroids, so new data
must remain inside the same valid domain.

!!! important

    Apply exactly the same preprocessing to training and inference data.

---

## `n_clusters`

The number of clusters is configured with:

```python
n_clusters=3
```

and the same value is used for every divergence.

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    local_model="ridge",
    combiner="mean",
    n_clusters=5,
)
```

This fits:

```text
Euclidean BregmanKMeans -> 5 clusters
GKL BregmanKMeans       -> 5 clusters
```

### Effect on the F-Step

If there are \(M\) divergences and \(K\) clusters, the next stage can fit up to

\[
M\times K
\]

local predictive models.

So increasing `n_clusters` increases model locality and model count.

---

## Choosing the number of clusters

A smaller value gives:

```text
fewer clusters
     ↓
more observations per cluster
     ↓
less-local predictive models
```

A larger value gives:

```text
more clusters
     ↓
fewer observations per cluster
     ↓
more-local predictive models
```

For classification, very large values can create clusters containing only one
class, which may make some local classifiers impossible to fit.

The current `KFCProcedure` does not automatically tune `n_clusters`.

Evaluate it externally if it is an important hyperparameter.

---

## Number of samples constraint

`BregmanKMeans.fit()` explicitly checks:

```python
if X.shape[0] < self.n_clusters:
    raise ValueError("n_samples < n_clusters")
```

Therefore:

\[
n_{\text{samples}}\geq n_{\text{clusters}}
\]

must hold for the internal K-Step training subset.

Remember that KFC uses only approximately half of the outer training dataset
for K-Step/F-Step fitting.

---

## How Bregman K-means works

Each divergence-specific model uses a Lloyd-style iterative algorithm.

For one initialization:

```text
initialize centroids
       │
       ▼
compute divergence to every centroid
       │
       ▼
assign each sample to nearest centroid
       │
       ▼
recompute centroids
       │
       ▼
measure distortion
       │
       ├── converged? -> stop
       │
       └── no -> repeat
```

The assignment step is:

\[
c_i
=
\operatorname*{arg\,min}_k
D_\phi(X_i,\mu_k).
\]

The implementation updates each cluster centroid using the arithmetic mean of
the assigned samples.

---

## Centroid update

For cluster \(k\), the implementation computes

\[
\mu_k
=
\frac{1}{|C_k|}
\sum_{i\in C_k}X_i.
\]

This arithmetic-mean update is used for every built-in divergence in the
current `BregmanKMeans` implementation.

---

## Empty clusters

During an iteration, a cluster may receive no samples.

The implementation handles this case by reinitializing that centroid with a
random training observation:

```python
if counts[k] == 0:
    centroids[k] = X[rng.randint(n)]
```

This prevents division-by-zero failures and lets the Lloyd iterations
continue.

---

## Initialization

`BregmanKMeans` chooses initial centroids by sampling data points without
replacement:

```python
idx = rng.choice(
    X.shape[0],
    self.n_clusters,
    replace=False,
)
```

The code comments refer to this as a k-means++-style initialization, but the
current implementation is specifically **uniform random sampling of training
observations**, not the probability-weighted k-means++ seeding algorithm.

!!! note "Current source behavior"

    Treat the initialization as random data-point initialization.

---

## Multiple initializations

`BregmanKMeans` has:

```python
n_init=10
```

by default.

For every initialization it runs the full Lloyd loop and retains the solution
with the smallest final average distortion.

Conceptually:

```text
initialization 1 -> distortion d1
initialization 2 -> distortion d2
...
initialization 10 -> distortion d10
                    │
                    ▼
             keep minimum
```

### K-Step limitation

`KStep` does not currently expose an `n_init` parameter.

It creates `BregmanKMeans` without overriding `n_init`, so each
divergence-specific clustering currently uses the internal default:

```python
n_init=10
```

---

## `max_iter`

Configure the maximum number of Lloyd iterations per initialization with:

```python
max_iter=300
```

Example:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mean",
    n_clusters=3,
    max_iter=500,
)
```

This value is forwarded by `KStep` to each `BregmanKMeans`.

---

## `tol`

The convergence tolerance defaults to:

```python
tol=1e-4
```

After each iteration, the implementation computes relative distortion change:

\[
\frac{
|d_{\text{previous}}-d_{\text{current}}|
}{
|d_{\text{previous}}|+10^{-12}
}.
\]

The run stops when this is below `tol`.

Example:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mean",
    tol=1e-6,
)
```

A smaller tolerance can require more iterations before convergence.

---

## Distortion and `inertia_`

For the current Bregman implementation, distortion is the average minimum
divergence to a centroid:

\[
\frac{1}{n}
\sum_{i=1}^{n}
\min_k D_\phi(X_i,\mu_k).
\]

The best run stores this value in:

```python
bregman_model.inertia_
```

Despite the familiar `inertia_` name, this value is not necessarily the
classical Euclidean sum of squared distances.

It is the selected model's **average Bregman distortion** in the current
implementation.

---

## Memory-safe distortion calculation

During fitting, distortion is calculated in blocks:

```python
block=4096
```

through `_distortion_stream()`.

This avoids allocating one very large sample-by-centroid distance matrix solely
for the distortion calculation.

The assignment step itself still computes the full distance matrix for the
current Lloyd iteration.

---

## `random_state`

Use:

```python
random_state=42
```

for reproducible centroid initialization and empty-cluster reinitialization.

```python
model = KFCClassifier(
    divergences=["euclidean"],
    local_model="logistic_regression",
    combiner="majority_vote",
    n_clusters=3,
    random_state=42,
)
```

The same K-Step random seed is passed to every divergence-specific
`BregmanKMeans`.

This means each clustering model uses reproducible pseudo-random
initializations when the same data and configuration are reused.

---

## Multiple divergences are fitted independently

The K-Step loops through:

```python
for div in self.divergences:
    ...
    model.fit(X)
```

There is no interaction between divergence-specific cluster models.

Therefore:

```text
Euclidean clustering ─┐
GKL clustering ───────┼── independent
Logistic clustering ──┤
IS clustering ────────┘
```

The outputs are combined only later through the F-Step and C-Step.

---

## Fitted attributes

After KFC has been fitted:

```python
model.kstep_
```

contains the fitted K-Step.

Its primary fitted attributes are:

| Attribute | Description |
| --- | --- |
| `models_` | dictionary of fitted `BregmanKMeans` models |
| `clusters_` | training cluster labels for each divergence |

---

## Inspect fitted models

```python
kstep = model.kstep_

print(kstep.models_.keys())
```

For string-based divergences, you may see:

```text
dict_keys([
    "euclidean",
    "gkl",
    "logistic",
    "is",
])
```

Inspect one fitted model:

```python
euclidean_model = kstep.models_["euclidean"]

print(euclidean_model.cluster_centers_)
print(euclidean_model.labels_)
print(euclidean_model.inertia_)
print(euclidean_model.n_iter_)
```

---

## Inspect training cluster assignments

```python
for divergence, labels in model.kstep_.clusters_.items():
    print(
        divergence,
        labels.shape,
        set(labels),
    )
```

Example:

```text
euclidean (500,) {0, 1, 2}
gkl       (500,) {0, 1, 2}
logistic  (500,) {0, 1, 2}
is        (500,) {0, 1, 2}
```

The labels are later used by the F-Step to fit one local predictive model per
cluster.

---

## Predict cluster assignments

After fitting:

```python
clusters = model.kstep_.predict(X_new)
```

returns a dictionary:

```python
{
    "euclidean": array([...]),
    "gkl": array([...]),
    "logistic": array([...]),
    "is": array([...]),
}
```

Each array has shape:

```text
(n_samples,)
```

Example:

```python
clusters = model.kstep_.predict(X_test)

for divergence, labels in clusters.items():
    print(
        divergence,
        labels[:10],
    )
```

---

## Distance to centroids

The underlying `BregmanKMeans` also implements:

```python
transform(X)
```

which returns the divergence from every sample to every centroid.

Example:

```python
km = model.kstep_.models_["euclidean"]

D = km.transform(X_test)

print(D.shape)
```

With three clusters:

```text
(n_test, 3)
```

and:

\[
D_{ik}
=
D_\phi(X_i,\mu_k).
\]

The predicted cluster is the minimum-distance column.

---

## Using a divergence object directly

`KStep` accepts either:

```text
string names
```

or instances of:

```python
BaseBregmanDivergence
```

Example:

```python
from kfc_procedure.core.clustering.divergences import (
    SquaredEuclidean,
)

div = SquaredEuclidean()

model = KFCRegressor(
    divergences=[div],
    local_model="ridge",
    combiner="mean",
)
```

For a divergence instance, `_resolve()` simply returns the object rather than
creating one through the factory.

---

## Naming of divergence objects

When a string is supplied, the K-Step dictionary key is simply the lower-case
string:

```python
"euclidean"
"gkl"
"logistic"
"is"
```

When a divergence object is supplied, the key comes from its `name` attribute
when available.

For the built-ins, object names include:

```text
SquaredEuclidean.name -> "Euclidean"
GKLDivergence.name    -> "GKL"
LogisticLoss.name     -> "Logit"
ItakuraSaito.name     -> "Ita"
```

so direct objects can lead to keys such as:

```text
euclidean
gkl
logit
ita
```

rather than the factory identifiers:

```text
euclidean
gkl
logistic
is
```

!!! important "Object names and `divergences_params`"

    If you pass divergence **instances**, the K-Step does not use
    `divergences_params` to reconfigure them.

    Configure the object itself before supplying it.

---

## `divergences_params`

When divergences are supplied by string, K-Step resolves each one as:

```python
params = self.divergences_params.get(name, {})

divergence = BregmanDivergenceFactory.create(
    div,
    **params,
)
```

Example:

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
    ],
    divergences_params={
        "euclidean": {
            "validate_domain": True,
        },
        "gkl": {
            "validate_domain": True,
        },
    },
    local_model="ridge",
    combiner="mean",
)
```

The built-in divergences currently have very few practical constructor
hyperparameters, so `divergences_params` becomes more useful when adding
custom divergence implementations.

---

## Custom divergence

A custom divergence must subclass:

```python
BaseBregmanDivergence
```

and implement at least:

```python
in_domain(...)
phi(...)
grad_phi(...)
```

A minimal structure is:

```python
import numpy as np

from kfc_procedure.core.clustering.divergences import (
    BaseBregmanDivergence,
)


class MyDivergence(BaseBregmanDivergence):

    name = "my_divergence"
    family = "custom"

    def in_domain(self, X):
        return np.isfinite(X).all()

    def phi(self, X):
        X = np.asarray(X, dtype=float)
        # return one generator value per sample
        ...

    def grad_phi(self, X):
        X = np.asarray(X, dtype=float)
        # return gradient with same feature dimension
        ...
```

Then pass an instance:

```python
model = KFCRegressor(
    divergences=[
        MyDivergence(),
    ],
    local_model="ridge",
    combiner="mean",
)
```

The base class provides a generic pairwise Bregman `distance()` implementation.

---

## Registering a custom divergence

For factory-based use, register a subclass with
`BregmanDivergenceFactory`.

Conceptually:

```python
from kfc_procedure.core.clustering.divergences import (
    BregmanDivergenceFactory,
)

@BregmanDivergenceFactory.register("my_divergence")
class MyDivergence(BaseBregmanDivergence):
    ...
```

Then it can be selected by name:

```python
model = KFCRegressor(
    divergences=["my_divergence"],
    local_model="ridge",
    combiner="mean",
)
```

---

## Current `verbose` behavior

`KStep` exposes:

```python
verbose=False
```

and `BregmanKMeans` itself also has a `verbose` argument that can print
iteration diagnostics such as:

```text
iter=12 distortion=0.123456
```

However, in the current source, `KStep.fit()` creates `BregmanKMeans` without
passing its own `verbose` value:

```python
model = BregmanKMeans(
    divergence=divergence,
    n_clusters=self.n_clusters,
    max_iter=self.max_iter,
    tol=self.tol,
    random_state=self.random_state,
)
```

Therefore:

```python
KStep(verbose=True)
```

does **not currently enable** the underlying Bregman K-means iteration output.

!!! note "Current implementation limitation"

    The `verbose` field exists on `KStep`, but is not forwarded to
    `BregmanKMeans` in the current source version.

---

## Parameter summary

| Parameter | Default | Effect |
| --- | ---: | --- |
| `divergences` | required | clustering geometries to fit |
| `divergences_params` | `{}` | constructor kwargs for string divergences |
| `n_clusters` | `3` | clusters per divergence |
| `max_iter` | `300` | maximum Lloyd iterations per initialization |
| `tol` | `1e-4` | relative distortion convergence tolerance |
| `verbose` | `False` | stored by KStep; not currently forwarded |
| `random_state` | `None` | centroid initialization reproducibility |

The underlying `BregmanKMeans` additionally uses:

```text
n_init=10
```

but K-Step does not currently expose this parameter.

---

## Recommended starting configuration

For arbitrary numeric inputs:

```python
model = KFCRegressor(
    divergences=["euclidean"],
    local_model="ridge",
    combiner="mean",
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=42,
)
```

For data safely scaled into \((0,1)\):

```python
model = KFCRegressor(
    divergences=[
        "euclidean",
        "gkl",
        "logistic",
        "is",
    ],
    local_model="ridge",
    combiner="weighted_mean",
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=42,
)
```

---

## Debugging K-Step

If fitting fails, check these items first.

### 1. Are all values finite?

```python
import numpy as np

print(np.isfinite(X_train).all())
```

Expected:

```text
True
```

---

### 2. Does the input satisfy the divergence domain?

For GKL or Itakura–Saito:

```python
print((X_train > 0).all())
```

For logistic:

```python
print(
    ((X_train > 0) & (X_train < 1)).all()
)
```

---

### 3. Is `n_clusters` too large?

The internal K-Step subset contains only about half of the outer training
data.

Ensure:

```text
number of K-Step samples >= n_clusters
```

and remember that useful local models generally need substantially more than
one sample per cluster.

---

### 4. Did every divergence fit?

After successful fitting:

```python
print(model.kstep_.models_.keys())
```

Compare the result with the divergence list you requested.

---

### 5. Inspect distortions

```python
for name, km in model.kstep_.models_.items():
    print(
        name,
        "distortion=",
        km.inertia_,
        "iterations=",
        km.n_iter_,
    )
```

Do not directly compare distortion magnitudes across fundamentally different
divergences as if they were on a common metric scale.

---

## K-Step and F-Step connection

The K-Step returns cluster labels:

```python
clusters = model.kstep_.predict(X)
```

The F-Step then uses those labels to route observations to their local models:

```text
divergence
    │
    ▼
cluster assignment
    │
    ▼
local model for that divergence/cluster
    │
    ▼
prediction
```

The K-Step does not itself make supervised predictions.

Its output is the routing structure used by the F-Step.

---

## Mental model

!!! quote ""

    **K-Step asks several clustering geometries to provide several plausible
    partitions of the same observations.**

\[
\boxed{
X
\rightarrow
\left\{
\begin{array}{c}
\text{Euclidean clustering}\\
\text{GKL clustering}\\
\text{Logistic clustering}\\
\text{Itakura-Saito clustering}
\end{array}
\right.
\rightarrow
\text{F-Step}
}
\]

