"""
Itakura-Saito divergence.

This module implements the Itakura-Saito divergence induced by the
generator

    phi(x) = -Σ log(x_i).

The resulting divergence is

    D(x, y)
    =
    Σ [x_i / y_i - log(x_i / y_i) - 1].

The divergence is scale invariant and is widely used in signal
processing, spectral analysis, audio source separation, and
nonnegative matrix factorization.

Classes
-------
ItakuraSaito
    Itakura-Saito Bregman divergence.

Domain
------
All input values must satisfy

    x_i > 0.

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

@BregmanDivergenceFactory.register("is")
class ItakuraSaito(BaseBregmanDivergence):
    """
    Itakura-Saito divergence.

    Generator  φ(x)     = −Σᵢ ln(xᵢ)
    Divergence D(x, y)  = Σᵢ [ xᵢ/yᵢ − ln(xᵢ/yᵢ) − 1 ]
    Gredient   ∇φ(x)    = −1/x

    Exponential family : Exponential / Gamma
    Domain             : (0, +∞)ᵈ
    """

    name = "Ita"
    family = "Exponential / Gamma"

    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return -np.sum(np.log(X), axis=1)

    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        return -1.0 / np.asarray(X, dtype=np.float64)
    
    def in_domain(self, X: np.ndarray) -> bool:
        return np.all(X > 0)
    
    def distance(self, X: np.ndarray, Y: np.ndarray, *, clip: bool = True) -> np.ndarray:
        invY = 1.0 / Y
        logX = np.log(X)
        logY = np.log(Y)

        term1 = X @ invY.T
        logX_sum = np.sum(logX, axis=1)[:, None]
        logY_sum = np.sum(logY, axis=1)[None, :]

        D = term1 - (logX_sum - logY_sum) - X.shape[1]
        
        if clip:
            np.maximum(D, 0.0, out=D)
        return D