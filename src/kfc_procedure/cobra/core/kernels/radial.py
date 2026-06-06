"""
Radial (Gaussian-like) Kernel.

This kernel is a simplified radial basis function (RBF)
that maps distance to similarity using exponential decay.

Formula:
    K(D) = exp(-D)

Commonly used in:
- kernel methods
- clustering
- similarity learning
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("radial", "gaussian", "rbf")
class RadialKernel(BaseKernel):
    """
    Radial basis kernel (simplified form).
    """

    requires_grad = True
    mode = "continuous"

    def __call__(self, D):
        return np.exp(-D)
