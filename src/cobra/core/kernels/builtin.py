"""
Built-in kernel functions for COBRA consensus weighting.

This module provides concrete implementations of ``BaseKernel`` used
to convert adapted distance matrices into similarity weights.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Kernel functions transform distance values into similarity scores
that determine how much each neighbor contributes to the final
prediction.

These kernels control:

- neighborhood sharpness
- smoothness of weighting
- robustness to noise
- locality of the estimator pool

Kernel types
------------

1. IndicatorKernel (hard kernel)

    Produces binary weights based on a distance threshold.

2. RBFKernel (Gaussian kernel)

    Smooth exponential decay based on squared distance.

3. LaplaceKernel

    Exponential decay based on absolute distance.

Design goal
-----------
These kernels provide interchangeable weighting strategies for:

- hard neighbor selection
- smooth probabilistic weighting
- robust ensemble aggregation

Examples
--------
>>> kernel = KernelFactory.create("rbf", gamma=0.5)
>>> weights = kernel(distance_matrix)
"""

from __future__ import annotations

import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    """
    Hard threshold kernel (indicator function).

    This kernel assigns binary weights based on whether distances
    are below a given threshold.

    Mathematical form
    -----------------
    K(D) = 1 if D < threshold else 0

    Parameters
    ----------
    threshold : float, default=0.5
        Maximum distance allowed to be considered a neighbor.

    Notes
    -----
    This kernel performs hard selection of neighbors and is often used
    in strict COBRA variants.

    Examples
    --------
    >>> kernel = IndicatorKernel(threshold=0.3)
    >>> weights = kernel(D)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize indicator kernel.

        Parameters
        ----------
        threshold : float, default=0.5
            Distance cutoff for neighbor selection.
        """
        super().__init__()
        self.threshold = threshold

    def __call__(self, D: np.ndarray) -> np.ndarray:
        """
        Compute binary kernel weights.

        Parameters
        ----------
        D : np.ndarray
            Distance matrix.

        Returns
        -------
        np.ndarray
            Binary weight matrix.
        """
        return (D < self.threshold).astype(float)


@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    """
    Radial Basis Function (Gaussian) kernel.

    This kernel applies smooth exponential decay to distances,
    favoring closer neighbors.

    Mathematical form
    -----------------
    K(D) = exp(-gamma × D)

    Parameters
    ----------
    gamma : float, default=1.0
        Controls decay rate (higher = sharper locality).

    Notes
    -----
    One of the most commonly used kernels in COBRA-style methods.

    Examples
    --------
    >>> kernel = RBFKernel(gamma=0.5)
    >>> weights = kernel(D)
    """

    def __init__(self, gamma: float = 1.0):
        """
        Initialize RBF kernel.

        Parameters
        ----------
        gamma : float, default=1.0
            Kernel bandwidth parameter.
        """
        super().__init__()
        self.gamma = gamma

    def __call__(self, D: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel weights.

        Returns
        -------
        np.ndarray
            Smooth similarity weights.
        """
        return np.exp(-self.gamma * D)


@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    """
    Laplace kernel (exponential L1 decay).

    This kernel applies exponential decay based on absolute distance,
    producing heavier tails than Gaussian kernels.

    Mathematical form
    -----------------
    K(D) = exp(-gamma × |D|)

    Parameters
    ----------
    gamma : float, default=1.0
        Decay control parameter.

    Notes
    -----
    More robust to outliers compared to Gaussian kernel.

    Examples
    --------
    >>> kernel = LaplaceKernel(gamma=0.8)
    >>> weights = kernel(D)
    """

    def __init__(self, gamma: float = 1.0):
        """
        Initialize Laplace kernel.

        Parameters
        ----------
        gamma : float, default=1.0
            Decay rate parameter.
        """
        super().__init__()
        self.gamma = gamma

    def __call__(self, D: np.ndarray) -> np.ndarray:
        """
        Compute Laplace kernel weights.

        Returns
        -------
        np.ndarray
            Exponentially decayed similarity matrix.
        """
        return np.exp(-self.gamma * np.abs(D))
