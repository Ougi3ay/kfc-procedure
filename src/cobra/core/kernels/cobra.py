"""
COBRA Kernel.

A threshold-based discrete kernel used in COBRA-style aggregation.

Formula:
    K(D) = 1 if D < threshold else 0

This kernel performs hard neighborhood selection.
"""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("cobra")
class COBRAKernel(BaseKernel):
    """
    Binary threshold kernel.

    Parameters
    ----------
    threshold : float, default=0.5
        Distance cutoff for similarity.
    """

    mode = "discrete"
    requires_grad = False

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def __call__(self, D):
        return (D < self.threshold).astype(float)
