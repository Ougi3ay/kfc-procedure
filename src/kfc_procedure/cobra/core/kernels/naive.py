"""
Naive Kernel.

Identity transformation kernel (no mapping).

K(D) = D

Used mainly for:
- debugging
- baseline comparison
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("naive")
class NaiveKernel(BaseKernel):
    """
    Identity kernel (no transformation).
    """

    mode = "discrete"
    requires_grad = False

    def __call__(self, D):
        return D
