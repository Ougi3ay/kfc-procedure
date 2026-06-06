"""
Biweight Kernel.

A compact kernel with smooth quadratic decay.

Formula:
    K(D) = (1 - D)^2  if D < 1
           0         otherwise
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("biweight")
class BiweightKernel(BaseKernel):
    """
    Biweight kernel (compact support).
    """

    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, (1.0 - D) ** 2, 0.0)
