"""
Epanechnikov Kernel.

A compact-support kernel used in non-parametric estimation.

Formula:
    K(D) = 1 - D   if D < 1
           0       otherwise

This kernel is not differentiable at boundary and is non-smooth.
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("epanechnikov")
class EpanechnikovKernel(BaseKernel):
    """
    Compact-support Epanechnikov kernel.
    """

    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, 1.0 - D, 0.0)
