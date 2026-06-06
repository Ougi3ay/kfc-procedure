"""
Two parameter kernel adapter for COBRA.
This adapter allows linear combinations of two distance matrices,
enablingflexible fusion of multiple distance representations in the COBRA pipeline.

The transformation is defined as:
    D' = alpha * X + beta * Y

Where:
- X is the first distance matrix (e.g., from metric 1)
- Y is the second distance matrix (e.g., from metric 2)
- alpha and beta are learnable or tunable parameters that 
control the contribution of each distance matrix to 
the final adapted distance used in kernel construction.
"""
from __future__ import annotations

import numpy as np
from .base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)

@KernelAdapterFactory.register("two_parameter")
class TwoParameterKernelAdapter(BaseKernelAdapter):
    """
    Linear combination kernel adapter.

    This adapter combines two distance matrices:

        D' = alpha * X + beta * Y

    This is useful for multi-metric COBRA setups where different
    distance representations are fused.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Weight for first distance matrix.

        beta : float, default=0.0
            Weight for second distance matrix.
        """
        super().__init__(alpha=alpha, beta=beta)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Combine one or two distance matrices.

        Parameters
        ----------
        *distances : np.ndarray
            - 1 matrix → alpha * X
            - 2 matrices → alpha * X + beta * Y

        Returns
        -------
        np.ndarray
            Combined distance matrix.

        Raises
        ------
        ValueError
            If more than 2 matrices are provided or shapes mismatch.
        """
        if len(distances) == 0:
            raise ValueError("At least 1 distance matrix required")

        x = distances[0]

        if len(distances) == 1:
            return self.alpha * x

        if len(distances) > 2:
            raise ValueError("Maximum 2 distance matrices allowed")

        y = distances[1]

        if x.shape != y.shape:
            raise ValueError("Distance matrices must have same shape")

        return self.alpha * x + self.beta * y