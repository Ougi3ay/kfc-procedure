"""
Cauchy Kernel.

A heavy-tailed kernel providing robust similarity decay.

Formula:
    K(D) = 1 / (1 + D)

Useful for:
- robust similarity modeling
- outlier-resistant weighting
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("cauchy")
class CauchyKernel(BaseKernel):
    """
    Cauchy kernel with heavy-tailed decay.
    """

    requires_grad = True
    mode = "continuous"

    def __call__(self, D):
        return 1.0 / (1.0 + D)
