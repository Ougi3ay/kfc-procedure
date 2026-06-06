"""
Triangular Kernel.

Simple linear compact kernel.

Formula:
    K(D) = 1 - |D|   if D < 1
           0         otherwise
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("triangular")
class TriangularKernel(BaseKernel):
    """
    Linear triangular kernel.
    """

    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, 1.0 - np.abs(D), 0.0)
