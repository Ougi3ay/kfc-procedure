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


@KernelAdapterFactory.register("gradientcobra")
class GradientCOBRAKernelAdapter(BaseKernelAdapter):
    """
    Kernel adapter for GradientCOBRA aggregation.

    This adapter applies a bandwidth scaling factor to a single
    distance matrix before kernel evaluation.

    Mathematical form
    -----------------
    adapted_distance = bandwidth × distance

    Parameters
    ----------
    bandwidth : float, default=1.0
        Scaling parameter controlling the effective neighborhood size
        in kernel aggregation.

    Notes
    -----
    This adapter expects exactly one distance matrix.

    Examples
    --------
    >>> adapter = GradientCOBRAKernelAdapter(bandwidth=2.0)
    >>> adapted = adapter.transform(distance_matrix)
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize GradientCOBRA kernel adapter.

        Parameters
        ----------
        bandwidth : float, default=1.0
            Distance scaling hyperparameter.
        """
        super().__init__(bandwidth=bandwidth)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Scale a single distance matrix using bandwidth.

        Parameters
        ----------
        *distances : np.ndarray
            Expected:
                distances[0] = prediction-space distance matrix

        Returns
        -------
        np.ndarray
            Scaled distance matrix for kernel computation.

        Raises
        ------
        ValueError
            If the number of provided distance matrices is not exactly one.

        Examples
        --------
        >>> adapter.transform(distance_matrix)
        """
        if len(distances) != 1:
            raise ValueError(
                "GradientCOBRA expects exactly 1 distance matrix"
            )

        return self.bandwidth * distances[0]


@KernelAdapterFactory.register("mixcobra")
class MixCOBRAKernelAdapter(BaseKernelAdapter):
    """
    Kernel adapter for MixCOBRA aggregation.

    This adapter combines input-space distance and prediction-space
    distance using weighted hyperparameters.

    Mathematical form
    -----------------
    adapted_distance = alpha × x_distance + beta × y_distance

    Parameters
    ----------
    alpha : float, default=1.0
        Weight applied to input-space distance.

    beta : float, default=0.0
        Weight applied to prediction-space distance.

    Notes
    -----
    This adapter supports:

    - one distance matrix -> alpha × x_distance
    - two distance matrices -> alpha × x_distance + beta × y_distance

    Examples
    --------
    >>> adapter = MixCOBRAKernelAdapter(alpha=1.0, beta=0.5)

    >>> adapted = adapter.transform(x_distance)

    >>> adapted = adapter.transform(
    ...     x_distance,
    ...     y_distance
    ... )
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        """
        Initialize MixCOBRA kernel adapter.

        Parameters
        ----------
        alpha : float, default=1.0
            Weight for input-space distance.

        beta : float, default=0.0
            Weight for prediction-space distance.
        """
        super().__init__(alpha=alpha, beta=beta)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Combine one or two distance matrices.

        Parameters
        ----------
        *distances : np.ndarray
            Expected:
                distances[0] = x_distance
                distances[1] = y_distance (optional)

        Returns
        -------
        np.ndarray
            Adapted distance matrix.

        Raises
        ------
        ValueError
            If no distance matrix is provided.

        ValueError
            If more than two distance matrices are provided.

        ValueError
            If x_distance and y_distance have different shapes.

        Examples
        --------
        >>> adapter.transform(x_distance)

        >>> adapter.transform(
        ...     x_distance,
        ...     y_distance
        ... )
        """
        if len(distances) == 0:
            raise ValueError(
                "At least one distance matrix is required"
            )

        if len(distances) > 2:
            raise ValueError(
                "MixCOBRA expects at most 2 distance matrices: "
                "(x_distance, y_distance)"
            )

        x_distance = distances[0]

        if len(distances) == 1:
            return self.alpha * x_distance

        y_distance = distances[1]

        if x_distance.shape != y_distance.shape:
            raise ValueError(
                "x_distance and y_distance must have the same shape"
            )

        return (
            self.alpha * x_distance
            + self.beta * y_distance
        )
