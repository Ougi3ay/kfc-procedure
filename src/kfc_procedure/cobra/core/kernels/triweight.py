"""
Triweight Kernel.

Higher-order compact kernel with smoother decay.

Formula:
    K(D) = (1 - D)^3  if D < 1
           0         otherwise
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("triweight")
class TriweightKernel(BaseKernel):
    """
    Triweight compact kernel.
    """

    mode = "compact"
    requires_grad = False

    def __call__(self, D):
        return np.where(D < 1.0, (1.0 - D) ** 3, 0.0)
