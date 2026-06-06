"""
One parameter kernel adapter.
This adapter applies a single scalar parameter to a distance matrix:
    D' = bandwidth * D
This is commonly used as a bandwidth control mechanism in kernel construction (e.g., Gaussian kernels).
The bandwidth parameter can be learned or tuned to optimize COBRA performance.
"""

from __future__ import annotations
import numpy as np
from .base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)

@KernelAdapterFactory.register("one_parameter")
class OneParameterKernelAdapter(BaseKernelAdapter):
    """
    Simple scaling kernel adapter.

    This adapter applies a single scalar parameter to a distance matrix:

        D' = bandwidth * D

    This is commonly used as a bandwidth control mechanism in kernel
    construction (e.g., Gaussian kernels).
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Parameters
        ----------
        bandwidth : float, default=1.0
            Scaling factor applied to the distance matrix.
        """
        super().__init__(bandwidth=bandwidth)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Scale a single distance matrix.

        Parameters
        ----------
        *distances : np.ndarray
            Exactly one distance matrix is required.

        Returns
        -------
        np.ndarray
            Scaled distance matrix.

        Raises
        ------
        ValueError
            If number of inputs is not exactly 1.
        """
        if len(distances) != 1:
            raise ValueError("Expected exactly 1 distance matrix")

        return self.bandwidth * distances[0]
