# Kernels

Kernels convert adapted COBRA distance matrices into **similarity or weight
matrices**.

They sit after the distance and adapter layers:

\[
\boxed{
\text{representation}
\rightarrow
\text{distance}
\rightarrow
\text{adapter}
\rightarrow
\text{kernel}
\rightarrow
\text{aggregation}
}
\]

The current source implements ten kernel classes under:

```text
kfc_procedure/cobra/core/kernels/
├── base.py
├── radial.py
├── exponential.py
├── reverse_cosh.py
├── cauchy.py
├── epanechnikov.py
├── biweight.py
├── triweight.py
├── triangular.py
├── naive.py
└── cobra.py
```

All kernels are registered through:

```python
KernelFactory
```

---

## Kernel layer vs bandwidth layer

A key design choice in the current source is that the kernel usually does
**not** own the COBRA bandwidth.

Instead, a kernel adapter transforms the raw distance matrix first.

For the one-parameter adapter:

\[
D'
=
hD,
\]

where:

```text
h = bandwidth
```

Then the kernel receives:

\[
D'.
\]

For example, the default RBF kernel computes:

\[
K(D')
=
e^{-D'}.
\]

So the effective default one-parameter COBRA weight is:

\[
K_h(D)
=
e^{-hD}.
\]

This is important when interpreting the package's bandwidth parameter.

---

## COBRA kernel pipeline

```mermaid
flowchart LR
    A["Representation"]
    D["Distance matrix D"]
    T["Kernel adapter"]
    DP["Adapted distance D'"]
    K["Kernel K(D')"]
    W["Similarity / weight matrix"]
    G["Aggregator"]

    A --> D --> T --> DP --> K --> W --> G
```

The distance determines the geometry.

The adapter inserts tunable parameters such as:

```text
bandwidth
alpha
beta
```

The kernel maps the adapted distances to weights.

---

# Built-in kernels

The current source registers:

| Registry name | Aliases | Class | Formula / behavior | Mode |
| --- | --- | --- | --- | --- |
| `radial` | `gaussian`, `rbf` | `RadialKernel` | \(e^{-D}\) | `continuous` |
| `exponential` | — | `ExponentialKernel` | \(e^{-D^p}\) | `continuous` |
| `reverse_cosh` | — | `ReverseCoshKernel` | \(\cosh(D)^{-p}\) | `continuous` |
| `cauchy` | — | `CauchyKernel` | \(1/(1+D)\) | `continuous` |
| `epanechnikov` | — | `EpanechnikovKernel` | \(1-D\) for \(D<1\), else 0 | `compact` |
| `biweight` | — | `BiweightKernel` | \((1-D)^2\) for \(D<1\), else 0 | `compact` |
| `triweight` | — | `TriweightKernel` | \((1-D)^3\) for \(D<1\), else 0 | `compact` |
| `triangular` | — | `TriangularKernel` | \(1-|D|\) for \(D<1\), else 0 | `compact` |
| `naive` | — | `NaiveKernel` | identity \(K(D)=D\) | `discrete` |
| `cobra` | — | `COBRAKernel` | 1 if \(D<t\), else 0 | `discrete` |

---

# `BaseKernel`

Every built-in kernel inherits from:

```python
BaseKernel
```

The base interface provides:

```python
__call__(D)
set_params(**params)
get_params(deep=True)
is_continuous()
is_discrete()
```

A custom kernel must implement:

```python
__call__(D)
```

which returns the transformed matrix.

---

## Base metadata

`BaseKernel` defines two class-level attributes:

```python
requires_grad = True
mode = "continuous"
```

Concrete kernels override these values where needed.

The source uses:

```text
requires_grad
```

to decide whether gradient-style optimization is allowed.

The source uses:

```text
mode
```

to describe the kernel behavior.

---

## Kernel modes

The source documents three modes:

```text
continuous
compact
discrete
```

The helper:

```python
is_continuous()
```

returns:

```python
self.mode == "continuous"
```

and:

```python
is_discrete()
```

returns:

```python
self.mode == "discrete"
```

!!! note "Compact kernels are a third category"

    A kernel with:

    ```python
    mode = "compact"
    ```

    returns `False` from both:

    ```python
    is_continuous()
    is_discrete()
    ```

    because the current helper methods test only exact string equality.

---

# Kernel parameters

`BaseKernel.__init__()` stores keyword parameters in:

```python
self.params
```

and also sets same-named attributes.

For example:

```python
kernel = ExponentialKernel(
    exponent=2.0,
)
```

stores:

```python
kernel.params == {
    "exponent": 2.0,
}
```

and:

```python
kernel.exponent == 2.0
```

---

## `get_params()`

```python
kernel.get_params()
```

returns a copy of the internal parameter dictionary.

The `deep` argument exists for scikit-learn-style compatibility but is not
otherwise used by the current implementation.

---

## `set_params()`

```python
kernel.set_params(
    exponent=3.0,
)
```

updates both:

```python
kernel.exponent
```

and:

```python
kernel.params["exponent"]
```

and returns the same object.

---

# `KernelFactory`

Create a registered kernel with:

```python
from kfc_procedure.cobra.core.kernels import (
    KernelFactory,
)

kernel = KernelFactory.create(
    "rbf"
)
```

Because `KernelFactory` inherits the package's `BaseFactory`, names are
normalized before lookup.

---

## RBF aliases

The radial kernel is registered as:

```python
@KernelFactory.register(
    "radial",
    "gaussian",
    "rbf",
)
```

Therefore all three names create:

```python
RadialKernel
```

The high-level COBRA estimators use:

```text
rbf
```

as their default kernel name.

---

# Radial / RBF kernel

The class is:

```python
RadialKernel
```

with registry names:

```text
radial
gaussian
rbf
```

Its formula is:

\[
K(D)
=
e^{-D}.
\]

The implementation is exactly:

```python
return np.exp(
    -D
)
```

---

## No `gamma` parameter in `RadialKernel`

The current `RadialKernel` constructor is inherited from `BaseKernel`.

Its implementation does not define or use:

```text
gamma
sigma
bandwidth
```

inside the kernel formula.

Bandwidth scaling is handled by the adapter layer.

So for the one-parameter adapter:

\[
D'
=
hD
\]

followed by:

\[
K(D')
=
e^{-D'}
\]

gives:

\[
K_h(D)
=
e^{-hD}.
\]

---

## Example

```python
import numpy as np

from kfc_procedure.cobra.core.kernels import (
    RadialKernel,
)

D = np.array([
    [0.0, 0.5],
    [1.0, 2.0],
])

kernel = RadialKernel()

K = kernel(D)

print(
    K
)
```

---

# Exponential kernel

The class is:

```python
ExponentialKernel
```

registered as:

```text
exponential
```

Its formula is:

\[
K(D)
=
\exp\left(
-D^{p}
\right),
\]

where:

```text
p = exponent
```

---

## Constructor

```python
ExponentialKernel(
    exponent=1.0,
)
```

The default:

```python
exponent=1.0
```

makes the formula identical to the radial kernel:

\[
e^{-D}.
\]

So with its default constructor:

```text
exponential
```

and:

```text
rbf
```

produce the same mathematical mapping.

---

## Change curvature

```python
kernel = ExponentialKernel(
    exponent=2.0,
)
```

then:

\[
K(D)
=
e^{-D^2}.
\]

Configure through a COBRA estimator with:

```python
kernel="exponential",
kernel_params={
    "exponent": 2.0,
}
```

---

# Reverse-cosh kernel

The class is:

```python
ReverseCoshKernel
```

registered as:

```text
reverse_cosh
```

Its formula is:

\[
K(D)
=
\frac{1}
{\cosh(D)^p},
\]

where:

```text
p = exponent
```

---

## Constructor

```python
ReverseCoshKernel(
    exponent=1.0,
)
```

The source describes this as a smooth kernel that strongly suppresses large
distances.

---

## Example configuration

```python
model = GradientCOBRA(
    kernel="reverse_cosh",
    kernel_params={
        "exponent": 2.0,
    },
)
```

---

# Cauchy kernel

The class is:

```python
CauchyKernel
```

registered as:

```text
cauchy
```

Its formula is:

\[
K(D)
=
\frac{1}{1+D}.
\]

The source describes it as a heavy-tailed kernel with robust similarity decay.

---

## Implementation

```python
return 1.0 / (
    1.0 + D
)
```

It has no specialized constructor parameters in the current source.

---

# Compact-support kernels

Four built-ins have:

```python
mode = "compact"
```

and:

```python
requires_grad = False
```

They produce exactly zero weight outside their support.

The current compact kernels are:

```text
epanechnikov
biweight
triweight
triangular
```

All use a cutoff at:

\[
D=1.
\]

---

# Epanechnikov kernel

The class is:

```python
EpanechnikovKernel
```

registered as:

```text
epanechnikov
```

The current source formula is:

\[
K(D)
=
\begin{cases}
1-D, & D<1,\\
0, & D\geq1.
\end{cases}
\]

The implementation is:

```python
np.where(
    D < 1.0,
    1.0 - D,
    0.0,
)
```

---

## Important naming detail

The implementation called `EpanechnikovKernel` is exactly the linear compact
mapping shown above.

This page preserves the package's source formula rather than substituting a
different textbook normalization or polynomial form.

---

# Biweight kernel

The class is:

```python
BiweightKernel
```

registered as:

```text
biweight
```

Its source formula is:

\[
K(D)
=
\begin{cases}
(1-D)^2, & D<1,\\
0, & D\geq1.
\end{cases}
\]

Implementation:

```python
np.where(
    D < 1.0,
    (1.0 - D) ** 2,
    0.0,
)
```

---

# Triweight kernel

The class is:

```python
TriweightKernel
```

registered as:

```text
triweight
```

Its source formula is:

\[
K(D)
=
\begin{cases}
(1-D)^3, & D<1,\\
0, & D\geq1.
\end{cases}
\]

Implementation:

```python
np.where(
    D < 1.0,
    (1.0 - D) ** 3,
    0.0,
)
```

---

# Triangular kernel

The class is:

```python
TriangularKernel
```

registered as:

```text
triangular
```

The implementation is:

```python
np.where(
    D < 1.0,
    1.0 - np.abs(D),
    0.0,
)
```

So the source formula is:

\[
K(D)
=
\begin{cases}
1-|D|, & D<1,\\
0, & D\geq1.
\end{cases}
\]

For ordinary non-negative distance matrices this reduces to:

\[
1-D
\]

inside the support.

---

## Source boundary condition

The condition used in the current implementation is:

```python
D < 1.0
```

not:

```python
abs(D) < 1.0
```

The absolute value is applied only inside:

```python
1.0 - abs(D)
```

This distinction normally does not matter for valid non-negative distances,
but it is the exact current source behavior.

---

# COBRA threshold kernel

The class is:

```python
COBRAKernel
```

registered as:

```text
cobra
```

It implements hard neighborhood selection.

Its formula is:

\[
K(D)
=
\begin{cases}
1, & D<t,\\
0, & D\geq t.
\end{cases}
\]

---

## Constructor

```python
COBRAKernel(
    threshold=0.5,
)
```

The default cutoff is:

```text
0.5
```

---

## Implementation

```python
return (
    D < self.threshold
).astype(float)
```

So the output is a floating-point binary matrix.

---

## Example

```python
from kfc_procedure.cobra.core.kernels import (
    COBRAKernel,
)

kernel = COBRAKernel(
    threshold=0.7,
)

K = kernel(D)
```

---

# Naive kernel

The class is:

```python
NaiveKernel
```

registered as:

```text
naive
```

It performs no similarity transformation:

\[
K(D)
=
D.
\]

The implementation simply returns:

```python
D
```

---

## Important interpretation

Unlike most other kernels, larger distances remain larger output values.

So if the output is later used as aggregation weights, the naive kernel does
not convert "near" into "large similarity."

The source describes it mainly as useful for:

```text
debugging
baseline comparison
```

---

## Source metadata

Despite being an identity mapping, the current class sets:

```python
mode = "discrete"
requires_grad = False
```

This page preserves that metadata exactly as implemented.

---

# `requires_grad`

The current continuous kernels set or inherit:

```python
requires_grad = True
```

These are:

```text
radial / rbf
exponential
reverse_cosh
cauchy
```

The compact and discrete kernels set:

```python
requires_grad = False
```

These are:

```text
epanechnikov
biweight
triweight
triangular
naive
cobra
```

---

# Why `requires_grad` matters

`GradientCOBRA` checks:

```python
if (
    method == "grad"
    and not self.kernel_.requires_grad
):
    method = "grid"
```

`MixCOBRARegressor` uses the same logic.

Therefore if:

```python
opt_method="grad"
```

is requested with a non-gradient kernel, the current source silently changes
the local optimization method to:

```text
grid
```

rather than continuing with gradient optimization.

---

## Example

Conceptually:

```python
model = GradientCOBRA(
    kernel="cobra",
    opt_method="grad",
)
```

resolves a kernel with:

```python
requires_grad == False
```

and the optimization method is switched internally to grid mode.

---

# Default kernels

The current defaults are:

| Estimator | Default kernel |
| --- | --- |
| `GradientCOBRA` | `rbf` |
| `MixCOBRARegressor` | `rbf` |
| `CombinedClassifier` | `rbf` |

Because:

```text
rbf
```

is an alias for:

```python
RadialKernel
```

all three default to:

\[
K(D)
=
e^{-D}
\]

after adapter transformation.

---

# Kernel parameters in COBRA estimators

The high-level estimators accept:

```python
kernel
```

and:

```python
kernel_params
```

For example:

```python
model = GradientCOBRA(
    kernel="exponential",
    kernel_params={
        "exponent": 2.0,
    },
)
```

The source resolves this with:

```python
KernelFactory.create(
    self.kernel,
    **(
        self.kernel_params
        or {}
    ),
)
```

---

# Inspect the resolved kernel

After component resolution or fitting:

```python
print(
    model.kernel_
)
```

Inspect its parameters:

```python
print(
    model.kernel_.get_params()
)
```

Inspect optimization metadata:

```python
print(
    model.kernel_.requires_grad
)

print(
    model.kernel_.mode
)
```

---

# Adapter scaling in GradientCOBRA

GradientCOBRA creates:

```python
OneParameterKernelAdapter(
    bandwidth=1.0,
)
```

through the factory.

During cross-validation:

```python
self.adapter_.set_params(
    bandwidth=bandwidth
)

D = self.adapter_.transform(
    self.distance_matrix_
)

K = self.kernel_(D)
```

Therefore candidate bandwidths act **before** the kernel.

---

# Effective RBF formula in GradientCOBRA

The adapter computes:

\[
D'
=
hD.
\]

The radial kernel computes:

\[
K(D')
=
e^{-D'}.
\]

Therefore:

\[
K_h(D)
=
e^{-hD}.
\]

!!! important "Bandwidth direction"

    In this source, increasing:

    ```text
    bandwidth
    ```

    multiplies the distance by a larger number before applying `exp(-D)`.

    So larger bandwidth values produce **faster decay and narrower effective
    neighborhoods**.

    This is the opposite parameter direction from conventions written as:

    \[
    K(D/h),
    \]

    where increasing \(h\) broadens the neighborhood.

---

# Effective compact-kernel support with bandwidth

The same adapter logic changes the support radius of compact kernels.

For example, Epanechnikov is positive when:

\[
D'
<
1.
\]

Since:

\[
D'=hD,
\]

the nonzero condition is:

\[
hD<1,
\]

or:

\[
D
<
\frac{1}{h}.
\]

Thus larger bandwidth values shrink the raw-distance support.

---

# Effective COBRA threshold with bandwidth

With:

```text
kernel="cobra"
```

the source applies:

\[
K(D')
=
1
\quad\text{if}\quad
D'<t.
\]

With:

\[
D'=hD,
\]

this becomes:

\[
D
<
\frac{t}{h}.
\]

Therefore both:

```text
threshold
bandwidth
```

influence the effective hard neighborhood radius.

---

# MixCOBRA two-parameter kernel input

In two-parameter MixCOBRA, the adapter combines two distance matrices:

\[
D'
=
\alpha D_X
+
\beta D_Y.
\]

Then the selected kernel receives this fused distance:

\[
K(
\alpha D_X+\beta D_Y
).
\]

With the default RBF kernel:

\[
K
=
\exp[
-(\alpha D_X+\beta D_Y)
].
\]

---

# MixCOBRA one-parameter kernel input

With:

```python
one_parameter=True
```

MixCOBRA first computes a distance on a concatenated representation and then
uses:

\[
D'
=
hD.
\]

The selected kernel is applied to that scaled distance exactly as in
GradientCOBRA.

---

# CombinedClassifier kernel flow

`CombinedClassifier` also creates:

```python
OneParameterKernelAdapter(
    bandwidth=1.0,
)
```

and during optimization:

```python
self.adapter_.set_params(
    bandwidth=bandwidth
)

D = self.adapter_.transform(
    self.distance_matrix_
)

K = self.kernel_(D)
```

So the same bandwidth-before-kernel interpretation applies.

---

# Kernel matrix shape

The kernel does not change matrix shape.

If:

```python
D.shape == (
    n_query,
    n_reference,
)
```

then:

```python
K = kernel(D)
```

normally has:

```python
K.shape == (
    n_query,
    n_reference,
)
```

The values are transformed element-wise by all current built-in kernels.

---

# Pairwise training kernel

During calibration and hyperparameter optimization, COBRA estimators often
start with a square distance matrix:

```text
(n_l, n_l)
```

The kernel transforms it to a square similarity/weight matrix of the same
shape.

Cross-validation then extracts blocks such as:

```python
K_val_train = K[
    np.ix_(
        val_idx,
        train_idx,
    )
]
```

for aggregation.

---

# Query-to-calibration kernel

At prediction time, a query-to-calibration distance matrix:

```text
(n_query, n_l)
```

is transformed into a kernel matrix of the same shape.

Each row then provides the calibration-sample weights used by the aggregator.

---

# Zero-weight rows

Compact and threshold kernels can easily produce rows containing only zeros.

For example, if every adapted distance is outside the compact support:

```text
D' >= 1
```

then:

```text
epanechnikov
biweight
triweight
triangular
```

produce zero for every calibration point.

Likewise, `cobra` produces all zeros when no distance is below its threshold.

The downstream aggregator/fitted estimator determines the fallback behavior.

---

# Regression fallback

For GradientCOBRA final prediction, the source passes:

```python
fallback=self.global_mean_
```

to the weighted-mean aggregator.

Thus if kernel weights have effectively zero total mass, regression can fall
back to the global calibration target mean.

Cross-validation code may use a different explicit fallback such as:

```python
fallback=0.0
```

inside its objective.

---

# Classification fallback

`CombinedClassifier` explicitly checks:

```python
if np.sum(w) <= 0:
    pred = self.global_majority_class_
```

during cross-validation prediction.

The final prediction path similarly has class-fallback logic in the wrapped
classifier.

This matters most for compact or hard-threshold kernels.

---

# Continuous vs compact behavior

A practical source-level distinction is:

```text
continuous kernels:
    normally assign a nonzero decaying value over broad distance ranges

compact kernels:
    assign exactly zero at D >= 1 after adaptation

discrete kernels:
    hard / special mappings in current source
```

The package does not enforce a universal probabilistic normalization on kernel
outputs.

They become raw weights for the configured aggregator.

---

# Kernel outputs are not normalized probabilities

For example, the RBF kernel may return:

```text
[0.95, 0.45, 0.02]
```

for one query row.

These values are not normalized to sum to one by the kernel itself.

A weighted aggregator can normalize them internally when computing a weighted
mean or vote.

---

# Direct kernel comparison

```python
import numpy as np

from kfc_procedure.cobra.core.kernels import (
    KernelFactory,
)


D = np.array([
    [0.0, 0.25, 0.75, 1.25],
])


for name in [
    "rbf",
    "cauchy",
    "epanechnikov",
    "biweight",
    "triweight",
    "triangular",
    "cobra",
    "naive",
]:
    kernel = KernelFactory.create(
        name
    )

    print(
        name,
        kernel(D),
    )
```

This uses the same adapted-distance values for every source implementation.

---

# Direct RBF with adapter

```python
from kfc_procedure.cobra.core.adapters import (
    OneParameterKernelAdapter,
)

from kfc_procedure.cobra.core.kernels import (
    RadialKernel,
)


adapter = OneParameterKernelAdapter(
    bandwidth=2.0,
)

kernel = RadialKernel()

D_adapted = adapter.transform(
    D
)

K = kernel(
    D_adapted
)
```

This computes:

\[
K
=
e^{-2D}.
\]

---

# Exponential vs RBF

With:

```python
ExponentialKernel(
    exponent=1.0,
)
```

the mapping is:

\[
e^{-D},
\]

which is the same as:

```python
RadialKernel()
```

The difference appears when:

```python
exponent != 1.
```

For example:

```python
exponent=2
```

produces:

\[
e^{-D^2}.
\]

---

# Compact-kernel comparison

For:

\[
0\leq D<1,
\]

the current source gives:

\[
K_{\text{Epan}}(D)
=
1-D,
\]

\[
K_{\text{Bi}}(D)
=
(1-D)^2,
\]

\[
K_{\text{Triweight}}(D)
=
(1-D)^3.
\]

As the power increases, values inside the support decay more strongly away
from zero.

At:

\[
D\geq1,
\]

all three are exactly zero.

For non-negative distances, the current triangular implementation also reduces
to:

\[
1-D
\]

inside the support, making it numerically equivalent to the current
Epanechnikov implementation on ordinary non-negative distance input.

!!! note "Current source equivalence"

    Because valid distance matrices are normally non-negative:

    ```text
    epanechnikov
    triangular
    ```

    currently produce the same values for ordinary COBRA distance inputs.

    Their code differs only through `abs(D)` inside the triangular branch.

---

# KernelFactory inspection

List registered names:

```python
from kfc_procedure.cobra.core.kernels import (
    KernelFactory,
)

print(
    KernelFactory.available()
)
```

The current source registers names including:

```text
biweight
cauchy
cobra
epanechnikov
exponential
gaussian
naive
radial
rbf
reverse_cosh
triangular
triweight
```

---

# Inspect one registration

```python
print(
    KernelFactory.info(
        "rbf"
    )
)
```

Because:

```text
rbf
gaussian
radial
```

are aliases, they resolve to the same registered class.

---

# Custom kernel

A custom kernel should subclass:

```python
BaseKernel
```

and implement:

```python
__call__(D)
```

Example:

```python
import numpy as np

from kfc_procedure.cobra.core.kernels import (
    BaseKernel,
    KernelFactory,
)


@KernelFactory.register(
    "inverse_square"
)
class InverseSquareKernel(
    BaseKernel
):

    requires_grad = True
    mode = "continuous"

    def __call__(
        self,
        D,
    ):
        D = np.asarray(D)

        return 1.0 / (
            1.0 + D ** 2
        )
```

Then:

```python
model = GradientCOBRA(
    kernel="inverse_square",
)
```

can resolve it through the normal factory path.

---

# Custom kernel with parameters

```python
@KernelFactory.register(
    "scaled_cauchy"
)
class ScaledCauchyKernel(
    BaseKernel
):

    requires_grad = True
    mode = "continuous"

    def __init__(
        self,
        power=1.0,
    ):
        super().__init__(
            power=power,
        )

    def __call__(
        self,
        D,
    ):
        return 1.0 / (
            1.0 + D
        ) ** self.power
```

Configure with:

```python
kernel="scaled_cauchy",
kernel_params={
    "power": 2.0,
}
```

---

# Choosing `requires_grad` for a custom kernel

The optimizer compatibility check trusts the kernel's:

```python
requires_grad
```

attribute.

Set:

```python
requires_grad = True
```

only when the kernel is intended to participate in the package's
gradient-style optimization path.

Set:

```python
requires_grad = False
```

for non-smooth, compact-threshold, or otherwise unsupported cases.

The current source does not derive this capability automatically from the
kernel formula.

---

# Custom kernel mode

Use one of the source's descriptive mode strings:

```text
continuous
compact
discrete
```

For example:

```python
mode = "compact"
```

for a bounded-support mapping.

Remember that the current convenience helpers only explicitly recognize:

```text
continuous
discrete
```

and have no separate:

```python
is_compact()
```

method.

---

# Debugging kernel configuration

## Check configured name

```python
print(
    model.kernel
)
```

---

## Check kernel parameters

```python
print(
    model.kernel_params
)
```

---

## Inspect the resolved object

```python
print(
    model.kernel_
)
```

---

## Inspect source metadata

```python
print(
    model.kernel_.mode
)

print(
    model.kernel_.requires_grad
)
```

---

## Inspect effective parameters

```python
print(
    model.kernel_.get_params()
)
```

---

## Inspect the adapter too

Because bandwidth is not normally stored in the RBF kernel itself:

```python
print(
    model.adapter_.get_params()
)
```

This is essential when interpreting the effective similarity rule.

---

# Debugging unexpected zero weights

If using a compact kernel:

```python
D = model.adapter_.transform(
    distance_matrix
)

K = model.kernel_(
    D
)

print(
    np.min(D),
    np.max(D),
)

print(
    np.sum(
        K > 0,
        axis=1,
    )
)
```

Rows with zero positive entries have no neighbors inside the current support.

---

# Debugging RBF decay

For the default RBF path:

```python
h = model.adapter_.bandwidth
```

and:

```python
D_adapted = h * distance_matrix
```

Then:

```python
K = np.exp(
    -D_adapted
)
```

Inspecting both matrices can reveal whether the optimized bandwidth makes the
kernel too concentrated or too broad.

---

# Current implementation summary

| Kernel | `requires_grad` | Mode | Parameters |
| --- | :---: | --- | --- |
| `radial` / `gaussian` / `rbf` | Yes | `continuous` | none |
| `exponential` | Yes | `continuous` | `exponent=1.0` |
| `reverse_cosh` | Yes | `continuous` | `exponent=1.0` |
| `cauchy` | Yes | `continuous` | none |
| `epanechnikov` | No | `compact` | none |
| `biweight` | No | `compact` | none |
| `triweight` | No | `compact` | none |
| `triangular` | No | `compact` | none |
| `naive` | No | `discrete` | none |
| `cobra` | No | `discrete` | `threshold=0.5` |

---

# Quick reference

| Goal | Source option |
| --- | --- |
| default smooth decay | `rbf` |
| alternate smooth power decay | `exponential` |
| heavy-tailed decay | `cauchy` |
| hyperbolic decay | `reverse_cosh` |
| compact linear support | `epanechnikov` / `triangular` |
| stronger compact decay | `biweight`, `triweight` |
| hard neighborhood | `cobra` |
| identity/debug mapping | `naive` |

---

# Mental model

!!! quote ""

    **The distance says how far two prediction representations are apart; the
    adapter decides how strongly that distance is scaled; the kernel turns the
    adapted distance into the weight used for consensus.**

\[
\boxed{
D
\rightarrow
D'
\rightarrow
K(D')
\rightarrow
\text{aggregation weight}
}
\]

For the default KFC/COBRA RBF path:

\[
\boxed{
D
\rightarrow
hD
\rightarrow
e^{-hD}
}
\]

so the optimized `bandwidth` belongs to the adapter rather than to the
`RadialKernel` implementation itself.

