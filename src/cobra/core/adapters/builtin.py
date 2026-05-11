"""
Built-in kernel adapter implementations for GradientCOBRA, MixCOBRA, etc.

This module provides concrete implementations of ``BaseKernelAdapter`` for
the main aggregation strategies used in the COBRA framework.

Implemented adapters
--------------------

1. GradientCOBRAKernelAdapter

    Used in GradientCOBRA where a single prediction-space distance matrix
    is scaled by a bandwidth hyperparameter before kernel evaluation.

    Logic:
        adapted_distance = bandwidth * distance

2. MixCOBRAKernelAdapter

    Used in MixCOBRA where both input-space distance and prediction-space
    distance are combined using weighted hyperparameters.

    Logic:
        adapted_distance = alpha * x_distance + beta * y_distance

These adapters are registered automatically using
``KernelAdapterFactory.register()`` and can be instantiated dynamically.

Examples
--------
>>> adapter = KernelAdapterFactory.create(
...     "gradientcobra",
...     bandwidth=2.0
... )

>>> adapter.transform(distance_matrix)

>>> adapter = KernelAdapterFactory.create(
...     "mixcobra",
...     alpha=1.0,
...     beta=0.5
... )

>>> adapter.transform(x_distance, y_distance)
"""

from __future__ import annotations

import numpy as np

from cobra.core.adapters.base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)
@KernelAdapterFactory.register("one_parameter")
class OneParameterKernelAdapter(BaseKernelAdapter):
    """adapted_distance = h × distance"""

    def __init__(self, bandwidth: float = 1.0):
        super().__init__(bandwidth=bandwidth)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        if len(distances) != 1:
            raise ValueError("Expected 1 distance matrix")
        return self.h * distances[0]


@KernelAdapterFactory.register("two_parameter")
class TwoParameterKernelAdapter(BaseKernelAdapter):
    """adapted_distance = alpha × x + beta × y"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        super().__init__(alpha=alpha, beta=beta)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        if len(distances) == 0:
            raise ValueError("At least 1 distance matrix required")

        x = distances[0]

        if len(distances) == 1:
            return self.alpha * x

        if len(distances) > 2:
            raise ValueError("Max 2 distance matrices allowed")

        y = distances[1]

        if x.shape != y.shape:
            raise ValueError("Shape mismatch")

        return self.alpha * x + self.beta * y
