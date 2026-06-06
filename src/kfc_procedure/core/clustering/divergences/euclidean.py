"""
Squared Euclidean Bregman divergence.

This module implements the squared Euclidean divergence, the canonical
Bregman divergence induced by the generator

    phi(x) = ||x||²₂.

The resulting divergence is

    D(x, y) = ||x - y||²₂.

This divergence corresponds to the Gaussian exponential family and is
the default distance measure used in k-means clustering.

Classes
-------
SquaredEuclidean
    Squared Euclidean Bregman divergence.

Notes
-----
Unlike other Bregman divergences, the squared Euclidean divergence
admits a highly optimized closed-form computation

    ||x - y||² = ||x||² - 2<x,y> + ||y||²

which can be implemented using a single BLAS matrix multiplication.

References
----------
Banerjee, A., Merugu, S., Dhillon, I. S., and Ghosh, J. (2005).
"Clustering with Bregman Divergences."
Journal of Machine Learning Research, 6, 1705-1749.
"""

from __future__ import annotations
import numpy as np
from .base import (
    BaseBregmanDivergence,
    BregmanDivergenceFactory
)

@BregmanDivergenceFactory.register("euclidean")
class SquaredEuclidean(BaseBregmanDivergence):
    """
    Squared Euclidean distance.

    Generator   φ(x)    = ‖x‖²₂  =  Σᵢ xᵢ²
    Divergence  D(x, y) = ‖x − y‖²₂
    Gradient    ∇φ(x)   = 2x

    Exponential family : Gaussian
    Domain             : ℝᵈ  (no restriction)

    Note :
    --------
    ``distance`` is overridden to use the identity 
        ‖x − y‖² = ‖x‖² − 2⟨x, y⟩ + ‖y‖²
    
    Which reduces the computation to a single matrix multiplication,
    avoiding the explicit (n, K, d) intermediate tensor create by
    the base-class einsum implementation.
    """

    name = "Euclidean"
    family = "Gaussian"

    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.sum(X * X, axis=1)

    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(X, dtype=np.float64)
    
    def in_domain(self, X: np.ndarray) -> bool:
        return True  # ℝᵈ
    
    def distance(self, X: np.ndarray, Y: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """
        Reduced distance computation for squared Euclidean:
            D(x, y) = ‖x‖² − 2⟨x, y⟩ + ‖y‖²
        via BLAS-Level matrix multiplication

        Uses ‖x−y‖² = ‖x‖² − 2 x·y + ‖y‖², O(nKd) with near-peak
        BLAS throughput.  Avoids the (n, K, d) broadcast tensor of the
        base-class einsum → lower peak memory.
        """

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        xx = np.sum(X * X, axis=1)[:, None]
        yy = np.sum(Y * Y, axis=1)[None, :]
        xy = X @ Y.T

        D = xx - 2.0 * xy + yy
        if clip:
            np.maximum(D, 0.0, out=D)
        return D