"""
Reverse Cosh Kernel.

This kernel applies a hyperbolic cosine transformation
inverted and controlled by an exponent parameter.

It is a smooth continuous kernel that strongly suppresses
large distances.

Formula:
    K(D) = 1 / (cosh(D) ^ exponent)

Used in:
- smooth similarity decay
- robust distance attenuation
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("reverse_cosh")
class ReverseCoshKernel(BaseKernel):
    """
    Reverse hyperbolic cosine kernel.

    Parameters
    ----------
    exponent : float, default=1.0
        Controls decay sharpness.
    """

    requires_grad = True
    mode = "continuous"

    def __init__(self, exponent: float = 1.0):
        super().__init__(exponent=exponent)

    def __call__(self, D):
        return 1.0 / (np.cosh(D) ** self.exponent)
