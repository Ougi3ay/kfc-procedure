"""
Exponential Kernel.

This kernel applies exponential decay over distance.

Formula:
    K(D) = exp(-D^exponent)

It is widely used in similarity learning and radial models.
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("exponential")
class ExponentialKernel(BaseKernel):
    """
    Exponential decay kernel.

    Parameters
    ----------
    exponent : float, default=1.0
        Controls curvature of decay.
    """

    requires_grad = True
    mode = "continuous"

    def __init__(self, exponent: float = 1.0):
        super().__init__(exponent=exponent)

    def __call__(self, D):
        return np.exp(-(D ** self.exponent))
