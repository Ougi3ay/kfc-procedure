"""
Kernel adapter package.

This package provides the adapter layer that connects the distance
computation stage to the kernel evaluation stage in the COBRA pipeline.

Kernel adapters are responsible for transforming raw distance matrices
by injecting tunable hyperparameters before kernel functions are applied.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Main responsibilities
---------------------

- apply bandwidth scaling
- combine multiple distance spaces
- inject optimization parameters
- prepare kernel-ready distances

Available components
--------------------

Base classes
^^^^^^^^^^^^

- ``BaseKernelAdapter``
    Abstract interface for all kernel adapters.

- ``KernelAdapterFactory``
    Registry-based factory for dynamic adapter creation.

Built-in implementations
^^^^^^^^^^^^^^^^^^^^^^^^

- ``GradientCOBRAKernelAdapter``
    Single-distance scaling using a bandwidth parameter.

- ``MixCOBRAKernelAdapter``
    Weighted combination of input-space and prediction-space distances.

Examples
--------
>>> from cobra.core.adapters import KernelAdapterFactory

>>> adapter = KernelAdapterFactory.create(
...     "mixcobra",
...     alpha=1.0,
...     beta=0.5
... )

>>> adapted_distance = adapter.transform(
...     x_distance,
...     y_distance
... )

Exports
-------
This package exposes the most commonly used adapter classes directly
for convenient imports.
"""

from .base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)

from .builtin import (
    GradientCOBRAKernelAdapter,
    MixCOBRAKernelAdapter,
)

__all__ = [
    "BaseKernelAdapter",
    "KernelAdapterFactory",
    "GradientCOBRAKernelAdapter",
    "MixCOBRAKernelAdapter",
]
