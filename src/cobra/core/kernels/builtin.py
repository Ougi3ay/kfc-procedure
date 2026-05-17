"""
Built-in kernel functions for COBRA consensus weighting.

This module provides concrete implementations of ``BaseKernel`` used
to convert adapted distance matrices into similarity weights.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Kernel functions transform distance values into similarity scores
that determine how much each neighbor contributes to the final
prediction.

These kernels control:

- neighborhood sharpness
- smoothness of weighting
- robustness to noise
- locality of the estimator pool

Design goal
-----------
These kernels provide interchangeable weighting strategies for:

- hard neighbor selection
- smooth probabilistic weighting
- robust ensemble aggregation

Examples
--------
>>> kernel = KernelFactory.create("rbf", gamma=0.5)
>>> weights = kernel(distance_matrix)
"""

from __future__ import annotations

from cycler import V
import numpy as np

from .base import BaseKernel, KernelFactory

@KernelFactory.register("reverse_cosh")
class ReverseCoshKernel(BaseKernel):
    requires_grad = True
    mode = "continuous"

    def __init__(self, exponent: float = 1.0):
        super().__init__(exponent=exponent)

    def __call__(self, D):
        return 1.0 / (np.cosh(D) ** self.exponent)

@KernelFactory.register("exponential")
class ExponentialKernel(BaseKernel):
    requires_grad = True
    mode = "continuous"
    def __init__(self, exponent: float = 1.0):
        super().__init__(exponent=exponent)

    def __call__(self, D):
        return np.exp(-(D ** self.exponent))

@KernelFactory.register("radial", "gaussian", "rbf")
class RadialKernel(BaseKernel):
    requires_grad = True
    mode = "continuous"
    def __call__(self, D):
        return np.exp(-D)

@KernelFactory.register("cauchy")
class CauchyKernel(BaseKernel):
    requires_grad = True
    mode = "continuous"
    def __call__(self, D):
        return 1.0 / (1.0 + D)

@KernelFactory.register("epanechnikov")
class EpanechnikovKernel(BaseKernel):
    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, 1.0 - D, 0.0)

@KernelFactory.register("biweight")
class BiweightKernel(BaseKernel):
    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, (1.0 - D) ** 2, 0.0)

@KernelFactory.register("triweight")
class TriweightKernel(BaseKernel):
    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, (1.0 - D) ** 3, 0.0)

@KernelFactory.register("triangular")
class TriangularKernel(BaseKernel):
    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, 1.0 - np.abs(D), 0.0)

@KernelFactory.register("naive")
class NaiveKernel(BaseKernel):
    mode = "discrete"
    requires_grad = False

    def __call__(self, D):
        return D

@KernelFactory.register("cobra")
class COBRAKernel(BaseKernel):

    mode = "discrete"
    requires_grad = False

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def __call__(self, D):
        return (D < self.threshold).astype(float)
