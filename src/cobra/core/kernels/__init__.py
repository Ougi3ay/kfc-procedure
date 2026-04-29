"""
Kernel package for COBRA influence weighting.

This package defines the kernel stage of the COBRA pipeline, where
distance matrices are transformed into similarity weights that
control the influence of each neighbor in the final prediction.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Kernels convert adapted distances into influence scores that determine
how strongly each estimator (or neighbor) contributes to the final
consensus prediction.

This stage is responsible for shaping:

- locality of predictions
- smoothness vs. sharpness of weighting
- robustness to noise and outliers
- effective neighborhood size

Design philosophy
-----------------
The kernel layer is designed to be:

- modular (swap kernels easily)
- extensible (add custom weighting functions)
- configurable (hyperparameters via factories)
- compatible with optimization pipelines

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseKernel``
    Abstract interface for all kernel functions.

Factory system
^^^^^^^^^^^^^^

- ``KernelFactory``
    Registry-based factory for dynamic kernel creation.

Built-in kernels
^^^^^^^^^^^^^^^^

- ``IndicatorKernel`` (hard kernel)
    Binary selection based on a distance threshold.

- ``RBFKernel`` (Gaussian kernel)
    Smooth exponential decay weighting.

- ``LaplaceKernel``
    Exponential decay based on absolute distance.

Examples
--------
>>> from cobra.core.kernels import KernelFactory

>>> kernel = KernelFactory.create("rbf", gamma=0.5)

>>> weights = kernel(distance_matrix)

Exports
-------
All commonly used kernel implementations are exposed for convenience
and pipeline integration.
"""

from .base import (
    BaseKernel,
    KernelFactory,
)

from .builtin import (
    IndicatorKernel,
    LaplaceKernel,
    RBFKernel,
)

__all__ = [
    "BaseKernel",
    "KernelFactory",
    "IndicatorKernel",
    "RBFKernel",
    "LaplaceKernel",
]