# Distances

Distance functions are a core part of the COBRA components in
`kfc_procedure`.

They define how two observations are compared once those observations have
been represented in the relevant COBRA space.

The current source provides five built-in distance implementations:

```text
euclidean
manhattan
minkowski
cosine
hamming
```

The implementation lives under:

```text
kfc_procedure/cobra/core/distances/
├── base.py
├── euclidean.py
├── manhattan.py
├── minkowski.py
├── cosine.py
└── hamming.py
```

These distances are used by:

```text
GradientCOBRA
MixCOBRARegressor
CombinedClassifier
```

through the shared:

```python
DistanceFactory
```

---

## Distance in the COBRA pipeline

The general flow is:

```mermaid
flowchart LR
    P["Prediction/input representation"]
    D["Pairwise distance matrix"]
    A["Kernel adapter"]
    K["Kernel weights"]
    G["Aggregator"]
    Y["Final prediction"]

    P --> D --> A --> K --> G --> Y
```

The distance step converts two matrices:

```python
x
y
```

into a pairwise matrix:

\[
D
\in
\mathbb{R}^{n_x\times n_y}.
\]

Each entry is:

\[
D_{ij}
=
d(x_i,y_j).
\]

---

# Built-in distances

| Registry name | Alias | Class | Main idea |
| --- | --- | --- | --- |
| `euclidean` | `l2` | `EuclideanDistance` | L2 distance |
| `manhattan` | `l1` | `ManhattanDistance` | L1 distance |
| `minkowski` | `lp` | `MinkowskiDistance` | generalized Lp distance |
| `cosine` | — | `CosineDistance` | angular dissimilarity |
| `hamming` | — | `HammingDistance` | fraction of mismatched coordinates |

---

# The distance interface

Every distance implementation inherits:

```python
BaseDistance
```

The abstract interface is:

```python
class BaseDistance(ABC):

    def __init__(
        self,
        **kwargs,
    ):
        ...

    def set_params(
        self,
        **params,
    ):
        ...

    def get_params(
        self,
        deep=True,
    ):
        ...

    @abstractmethod
    def matrix(
        self,
        x,
        y,
    ):
        ...
```

The only required computational method is:

```python
matrix(x, y)
```

---

## Expected matrix shapes

If:

```python
x.shape == (
    n_x,
    n_features,
)
```

and:

```python
y.shape == (
    n_y,
    n_features,
)
```

then:

```python
distance.matrix(
    x,
    y,
).shape
```

is expected to be:

```text
(n_x, n_y)
```

---

# `BaseDistance` parameters

The base constructor stores every keyword argument in:

```python
self.params
```

and also creates same-named attributes.

Conceptually:

```python
distance = SomeDistance(
    alpha=2.0,
)
```

results in:

```python
distance.params == {
    "alpha": 2.0,
}
```

and:

```python
distance.alpha == 2.0
```

---

## `get_params()`

```python
distance.get_params()
```

returns a copy of:

```python
self.params
```

The signature includes:

```python
deep=True
```

for scikit-learn-style compatibility, but the current implementation does not
use `deep`.

---

## `set_params()`

```python
distance.set_params(
    alpha=3.0,
)
```

updates both:

```python
distance.alpha
```

and:

```python
distance.params["alpha"]
```

and returns:

```python
self
```

for chaining.

---

# `DistanceFactory`

Distances are registered with:

```python
DistanceFactory
```

which subclasses the package's:

```python
BaseFactory
```

You can create a distance by name:

```python
from kfc_procedure.cobra.core.distances import (
    DistanceFactory,
)

distance = DistanceFactory.create(
    "euclidean"
)
```

---

## Aliases

The current registrations are:

```python
@DistanceFactory.register(
    "euclidean",
    "l2",
)
```

```python
@DistanceFactory.register(
    "manhattan",
    "l1",
)
```

```python
@DistanceFactory.register(
    "minkowski",
    "lp",
)
```

while:

```text
cosine
hamming
```

have one registered name each.

---

## Registry normalization

Because `DistanceFactory` inherits `BaseFactory`, registry names are:

```text
stripped
lowercased
```

before lookup.

So:

```python
DistanceFactory.create(
    "EUCLIDEAN"
)
```

resolves the same normalized key as:

```text
euclidean
```

---

# Euclidean distance

The implementation class is:

```python
EuclideanDistance
```

registered as:

```text
euclidean
l2
```

The source defines:

\[
d(x,y)
=
\sqrt{
\|x-y\|_2^2
}.
\]

Equivalent form:

\[
d(x,y)
=
\sqrt{
\sum_k
(x_k-y_k)^2
}.
\]

---

## Implementation

The source uses the identity:

\[
\|x-y\|^2
=
\|x\|^2
+
\|y\|^2
-
2x^\top y.
\]

The code is:

```python
x2 = np.sum(
    x ** 2,
    axis=1,
    keepdims=True,
)

y2 = np.sum(
    y ** 2,
    axis=1,
    keepdims=True,
).T

xy = x @ y.T

dist = np.sqrt(
    np.maximum(
        x2 + y2 - 2 * xy,
        0.0,
    )
)
```

---

## Numerical clipping

Before the square root, the source applies:

```python
np.maximum(
    ...,
    0.0,
)
```

This prevents tiny floating-point errors from producing negative squared
distances.

---

## Example

```python
import numpy as np

from kfc_procedure.cobra.core.distances import (
    EuclideanDistance,
)

x = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
])

y = np.array([
    [1.0, 0.0],
])

distance = EuclideanDistance()

D = distance.matrix(
    x,
    y,
)

print(
    D
)
```

The result has shape:

```text
(2, 1)
```

---

# Manhattan distance

The implementation class is:

```python
ManhattanDistance
```

registered as:

```text
manhattan
l1
```

The source defines:

\[
d(x,y)
=
\sum_k
|x_k-y_k|.
\]

---

## Implementation backend

The current source delegates to:

```python
scipy.spatial.distance_matrix
```

with:

```python
p=1
```

The implementation is:

```python
return distance_matrix(
    x,
    y,
    p=1,
)
```

---

## Example

```python
from kfc_procedure.cobra.core.distances import (
    ManhattanDistance,
)

distance = ManhattanDistance()

D = distance.matrix(
    x,
    y,
)
```

---

# Minkowski distance

The implementation class is:

```python
MinkowskiDistance
```

registered as:

```text
minkowski
lp
```

It generalizes L1 and L2 distances:

\[
d_p(x,y)
=
\left(
\sum_k
|x_k-y_k|^p
\right)^{1/p}.
\]

---

## Constructor

```python
MinkowskiDistance(
    p=3,
    **kwargs,
)
```

The default is:

```text
p = 3
```

This is a source-specific default.

---

## Special cases

The source documentation identifies:

\[
p=1
\]

as Manhattan distance and:

\[
p=2
\]

as Euclidean distance.

---

## Implementation

The current implementation constructs pairwise coordinate differences through
broadcasting:

```python
x[:, None, :]
-
y[None, :, :]
```

then computes:

```python
np.sum(
    np.abs(...) ** p,
    axis=2,
) ** (1 / p)
```

---

## Example

```python
from kfc_procedure.cobra.core.distances import (
    MinkowskiDistance,
)

distance = MinkowskiDistance(
    p=4,
)

D = distance.matrix(
    x,
    y,
)
```

---

## `distance_params`

Because `MinkowskiDistance` exposes:

```python
p
```

through its constructor, COBRA estimators can configure it using:

```python
distance_params
```

For example:

```python
model = GradientCOBRA(
    distance="minkowski",
    distance_params={
        "p": 4,
    },
)
```

The source passes:

```python
**distance_params
```

directly into:

```python
DistanceFactory.create(...)
```

---

# Cosine distance

The implementation class is:

```python
CosineDistance
```

registered as:

```text
cosine
```

The source defines cosine similarity as:

\[
s(x,y)
=
\frac{
x^\top y
}{
\|x\|
\|y\|
}
\]

and cosine distance as:

\[
d(x,y)
=
1-s(x,y).
\]

---

## Implementation

The code computes row norms:

```python
x_norm = np.linalg.norm(
    x,
    axis=1,
    keepdims=True,
)

y_norm = np.linalg.norm(
    y,
    axis=1,
    keepdims=True,
).T
```

then:

```python
sim = (
    x @ y.T
) / (
    x_norm * y_norm
    + 1e-12
)
```

and returns:

```python
1.0 - sim
```

---

## Zero-vector protection

The denominator contains:

```python
1e-12
```

to avoid division by zero.

This means the implementation does not raise simply because one of the vectors
has norm zero.

---

## Range noted by the source

The module notes that cosine distance is theoretically in:

\[
[0,2].
\]

It also notes that values are typically in:

\[
[0,1]
\]

for non-negative data.

---

## Example

```python
from kfc_procedure.cobra.core.distances import (
    CosineDistance,
)

distance = CosineDistance()

D = distance.matrix(
    x,
    y,
)
```

---

# Hamming distance

The implementation class is:

```python
HammingDistance
```

registered as:

```text
hamming
```

The source defines normalized Hamming distance as:

\[
d(x,y)
=
\frac{1}{d}
\sum_{k=1}^{d}
\mathbf{1}
[x_k\neq y_k].
\]

So the output measures the fraction of coordinates that differ.

---

## Output range

Because the distance is normalized by the number of features:

\[
0
\leq
d(x,y)
\leq
1.
\]

---

## Intended data type

The source explicitly describes Hamming distance as intended for:

```text
discrete
binary
categorical
symbolic
```

representations.

It also states that continuous features should be discretized before use.

---

# Numba implementation

The primary Hamming path uses:

```python
@nb.jit(
    nopython=True,
    parallel=True,
    fastmath=True,
)
```

for:

```python
hamming_matrix_numba()
```

The function loops over all sample pairs and all feature coordinates.

For every pair:

```python
if x[i, k] != y[j, k]:
    diff_count += 1.0
```

then:

```python
distances[i, j] = (
    diff_count
    / n_features
)
```

---

## NumPy fallback

The public `matrix()` method wraps the Numba path in:

```python
try:
    ...
except Exception:
    ...
```

If the Numba path fails, it uses:

```python
np.mean(
    x[:, None, :]
    !=
    y[None, :, :],
    axis=2,
)
```

This produces the same normalized mismatch fraction.

---

# Hamming and CombinedClassifier

The default distance for:

```python
CombinedClassifier
```

is:

```text
hamming
```

This aligns with the current prediction-space representation used by
`CombinedClassifier`, which consists of hard class predictions such as:

```text
[0, 1, 1, 0]
```

For two prediction vectors:

```text
[0, 1, 1, 0]
[0, 0, 1, 0]
```

one of four coordinates differs, so:

\[
d_{\text{Hamming}}
=
\frac14.
\]

---

# Distance defaults by estimator

The current source defaults are:

| Estimator | Default distance |
| --- | --- |
| `GradientCOBRA` | `euclidean` |
| `MixCOBRARegressor` | `euclidean` |
| `CombinedClassifier` | `hamming` |

These defaults reflect the representations used by the corresponding
estimators.

---

# Distance configuration

Each high-level COBRA estimator accepts:

```python
distance
```

and:

```python
distance_params
```

For example:

```python
model = GradientCOBRA(
    distance="cosine",
)
```

or:

```python
model = GradientCOBRA(
    distance="minkowski",
    distance_params={
        "p": 4,
    },
)
```

---

# How distance objects are resolved

The shared resolver contains:

```python
return DistanceFactory.create(
    distance,
    **(
        distance_params
        or {}
    ),
)
```

The high-level estimators use equivalent factory construction.

Therefore:

```text
distance
```

selects the registered implementation and:

```text
distance_params
```

are forwarded to its constructor.

---

# Inspect the fitted distance object

After fitting a COBRA estimator:

```python
print(
    model.distance_
)
```

To inspect parameters:

```python
print(
    model.distance_.get_params()
)
```

For Minkowski:

```python
print(
    model.distance_.p
)
```

---

# Prediction-space distance

For GradientCOBRA, the distance is computed on normalized prediction vectors.

During fitting:

```python
self.distance_matrix_ = self.distance_.matrix(
    self.Y_l_norm_,
    self.Y_l_norm_,
)
```

At prediction time:

```python
distance_matrix = self.distance_.matrix(
    Y_norm,
    self.Y_l_norm_,
)
```

So the distance object itself is independent of COBRA's normalization step.

It receives whatever representation the estimator passes into it.

---

# CombinedClassifier distance

CombinedClassifier stores its calibration prediction representation as:

```python
pred_l_
```

and computes:

```python
self.distance_matrix_ = self.distance_.matrix(
    self.pred_l_,
    self.pred_l_,
)
```

New prediction patterns are compared with:

```python
self.distance_.matrix(
    preds_space,
    self.pred_l_,
)
```

No separate prediction normalization is applied by the current
CombinedClassifier before this distance calculation.

---

# MixCOBRA distances

MixCOBRA can maintain two separate distance matrices.

In two-parameter mode:

```python
self.distance_matrix_x_
```

is computed on:

```python
X_l_norm_
```

and:

```python
self.distance_matrix_y_
```

is computed on:

```python
Y_l_norm_
```

The same configured distance implementation is used for both spaces.

---

## Two-space combination

The distance matrices are later combined by the two-parameter adapter as:

\[
D_{\text{mix}}
=
\alpha D_X
+
\beta D_Y.
\]

The distance class itself does not know about:

```text
alpha
beta
```

Those parameters belong to the adapter layer.

---

# One-parameter MixCOBRA

With:

```python
one_parameter=True
```

the source first concatenates:

```python
X_l_norm_
Y_l_norm_
```

into:

```python
mix_features_
```

Then the configured distance is computed once on the combined representation.

So:

```text
distance
```

still controls geometry, while the one-parameter adapter later scales the
result.

---

# Distance vs kernel

Distance and kernel are separate concepts.

The distance computes:

\[
D
=
d(x,y).
\]

The adapter may rescale it, for example:

\[
D'
=
hD.
\]

Then the kernel turns that value into a similarity weight:

\[
K(D').
\]

So choosing:

```python
distance="euclidean"
```

does not by itself define the final aggregation weight.

---

# Example with RBF kernel

Suppose Euclidean distance produces:

\[
D=2.
\]

If the adapter applies:

\[
h=0.5,
\]

then:

\[
D'=1.
\]

The RBF kernel then receives:

```text
1
```

rather than the original:

```text
2.
```

This separation is why distance configuration and bandwidth optimization are
handled independently in the source.

---

# Euclidean vs Minkowski `p=2`

The source provides both:

```text
euclidean
```

and:

```text
minkowski with p=2
```

These represent the same L2 distance mathematically.

However, the implementation paths differ.

`EuclideanDistance` uses the dot-product identity:

```text
||x||² + ||y||² - 2<x,y>
```

while `MinkowskiDistance` uses explicit pairwise broadcasting.

Therefore they are not implemented identically even when:

```python
p=2
```

---

# Manhattan vs Minkowski `p=1`

Similarly:

```text
manhattan
```

and:

```text
minkowski with p=1
```

represent the same L1 formula mathematically.

But:

```text
ManhattanDistance
```

delegates to SciPy's:

```python
distance_matrix(..., p=1)
```

while:

```text
MinkowskiDistance
```

uses NumPy broadcasting.

---

# Memory behavior

The implementations differ in how they form pairwise distances.

`EuclideanDistance` avoids a full:

```text
(n_x, n_y, n_features)
```

difference tensor.

`ManhattanDistance` delegates to SciPy.

`MinkowskiDistance` explicitly creates the broadcasted expression:

```python
x[:, None, :]
-
y[None, :, :]
```

which conceptually has three dimensions.

`HammingDistance` uses a Numba loop first, avoiding the broadcasted tensor in
its primary path, with a broadcasted NumPy fallback.

The source does not contain an automatic memory-based distance selector.

---

# Shape compatibility

The current distance implementations assume compatible feature dimensions.

For example:

```python
x.shape[1]
```

and:

```python
y.shape[1]
```

must represent the same coordinate dimension.

The classes do not define a common explicit manual check for mismatched
feature counts.

Incompatible shapes therefore fail through the underlying NumPy/SciPy
operations.

---

# Input validation scope

The built-in distance classes mainly perform:

```python
np.asarray(...)
```

before calculation.

They do not define a shared validation layer for:

```text
NaN
infinity
empty arrays
dtype compatibility
feature-count agreement
```

Those conditions are therefore handled implicitly by the mathematical
operations or by later stages.

---

# Custom distance

A custom metric can subclass:

```python
BaseDistance
```

and implement:

```python
matrix()
```

For example:

```python
import numpy as np

from kfc_procedure.cobra.core.distances import (
    BaseDistance,
    DistanceFactory,
)


@DistanceFactory.register(
    "squared_l2"
)
class SquaredL2Distance(
    BaseDistance
):

    def matrix(
        self,
        x,
        y,
    ):
        x = np.asarray(x)
        y = np.asarray(y)

        diff = (
            x[:, None, :]
            -
            y[None, :, :]
        )

        return np.sum(
            diff ** 2,
            axis=2,
        )
```

Then:

```python
distance = DistanceFactory.create(
    "squared_l2"
)
```

can resolve it.

---

# Custom distance with parameters

Because `BaseDistance` already supports arbitrary keyword parameters, a custom
distance can expose them through the constructor.

```python
@DistanceFactory.register(
    "scaled_l1"
)
class ScaledL1Distance(
    BaseDistance
):

    def __init__(
        self,
        scale=1.0,
    ):
        super().__init__(
            scale=scale,
        )

    def matrix(
        self,
        x,
        y,
    ):
        x = np.asarray(x)
        y = np.asarray(y)

        return self.scale * np.sum(
            np.abs(
                x[:, None, :]
                -
                y[None, :, :]
            ),
            axis=2,
        )
```

Then:

```python
model = GradientCOBRA(
    distance="scaled_l1",
    distance_params={
        "scale": 2.0,
    },
)
```

works through the normal factory path.

---

# Registering aliases

The factory supports multiple names:

```python
@DistanceFactory.register(
    "scaled_l1",
    "my_l1",
)
class ScaledL1Distance(
    BaseDistance
):
    ...
```

Both names resolve to the same class.

---

# Duplicate registration

Because `DistanceFactory` inherits `BaseFactory`, duplicate normalized names are
rejected.

For example, registering both:

```text
MyMetric
mymetric
```

in the same call is considered a duplicate after normalization.

Existing registry conflicts raise:

```python
KeyError
```

---

# Inspect available distances

```python
from kfc_procedure.cobra.core.distances import (
    DistanceFactory,
)

print(
    DistanceFactory.available()
)
```

The current built-ins include aliases, so the list should include names such
as:

```text
cosine
euclidean
hamming
l1
l2
lp
manhattan
minkowski
```

subject to normal module loading.

---

# Inspect one registration

```python
print(
    DistanceFactory.info(
        "euclidean"
    )
)
```

This uses the common factory metadata interface inherited from `BaseFactory`.

---

# Direct comparison example

```python
import numpy as np

from kfc_procedure.cobra.core.distances import (
    DistanceFactory,
)

x = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([
    [0.0, 1.0],
])

for name in [
    "euclidean",
    "manhattan",
    "minkowski",
    "cosine",
]:
    distance = DistanceFactory.create(
        name
    )

    print(
        name,
        distance.matrix(
            x,
            y,
        ),
    )
```

This compares the source implementations on the same coordinate
representation.

---

# Hamming example

```python
x = np.array([
    [0, 1, 1, 0],
    [1, 1, 0, 0],
])

y = np.array([
    [0, 0, 1, 0],
])

distance = DistanceFactory.create(
    "hamming"
)

D = distance.matrix(
    x,
    y,
)

print(
    D
)
```

For the first row, one of four coordinates differs:

\[
d=\frac14.
\]

---

# Choosing a distance

The source exposes different geometries rather than enforcing one globally.

A source-aligned summary is:

<div class="grid cards" markdown>

-   :material-ruler:{ .lg .middle } **Euclidean**

    ---

    Standard L2 geometry.

    Registry:

    ```text
    euclidean
    l2
    ```

-   :material-axis-arrow:{ .lg .middle } **Manhattan**

    ---

    Coordinate-wise absolute differences.

    Registry:

    ```text
    manhattan
    l1
    ```

-   :material-tune-variant:{ .lg .middle } **Minkowski**

    ---

    Generalized Lp distance with configurable:

    ```text
    p
    ```

-   :material-angle-acute:{ .lg .middle } **Cosine**

    ---

    Compares vector direction through angular dissimilarity.

-   :material-format-list-checks:{ .lg .middle } **Hamming**

    ---

    Counts normalized coordinate mismatches.

    Used by default in `CombinedClassifier`.

</div>

---

# Debugging distance configuration

## Check the configured name

```python
print(
    model.distance
)
```

---

## Check parameters

```python
print(
    model.distance_params
)
```

---

## Check the resolved object

After fitting:

```python
print(
    model.distance_
)
```

---

## Check effective parameters

```python
print(
    model.distance_.get_params()
)
```

---

## Check matrix shape

```python
D = model.distance_.matrix(
    A,
    B,
)

print(
    D.shape
)
```

Expected:

```text
(A.shape[0], B.shape[0])
```

---

## Check symmetry carefully

A metric such as Euclidean satisfies:

\[
d(x,y)=d(y,x).
\]

But the returned matrices:

```python
distance.matrix(
    A,
    B,
)
```

and:

```python
distance.matrix(
    B,
    A,
)
```

have transposed shapes.

To compare them:

```python
np.allclose(
    distance.matrix(
        A,
        B,
    ),
    distance.matrix(
        B,
        A,
    ).T,
)
```

---

# Current implementation summary

| Distance | Formula | Constructor parameter | Implementation |
| --- | --- | --- | --- |
| Euclidean | \(\sqrt{\sum (x-y)^2}\) | none | vectorized dot-product identity |
| Manhattan | \(\sum |x-y|\) | none | SciPy `distance_matrix(..., p=1)` |
| Minkowski | \((\sum |x-y|^p)^{1/p}\) | `p=3` | NumPy broadcasting |
| Cosine | \(1-\frac{x^\top y}{\|x\|\|y\|}\) | none | vectorized norms/dot products |
| Hamming | mean coordinate mismatch | none | Numba, NumPy fallback |

---

# Quick reference

| Registry | Alias | Typical current use |
| --- | --- | --- |
| `euclidean` | `l2` | default GradientCOBRA / MixCOBRA |
| `manhattan` | `l1` | alternative L1 geometry |
| `minkowski` | `lp` | configurable Lp geometry |
| `cosine` | — | angular comparison |
| `hamming` | — | default CombinedClassifier prediction-space comparison |

---

# Mental model

!!! quote ""

    **The distance layer decides what “nearby” means before the kernel turns
    that geometry into aggregation weights.**

\[
\boxed{
\text{representation}
\rightarrow
d(x,y)
\rightarrow
\text{distance matrix}
\rightarrow
\text{kernel}
\rightarrow
\text{aggregation}
}
\]

Distance selection changes the geometry of the COBRA neighborhood, while the
kernel and optimizer determine how that geometry is converted into prediction
weights.

