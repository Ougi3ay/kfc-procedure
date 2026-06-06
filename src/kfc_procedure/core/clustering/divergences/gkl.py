"""
Generalized Kullback-Leibler divergence.

This module implements the generalized Kullback-Leibler (GKL)
divergence, also known as the I-divergence, induced by the generator

    phi(x) = Σ x_i log(x_i) - x_i.

The resulting divergence is

    D(x, y)
    =
    Σ [x_i log(x_i / y_i) - (x_i - y_i)].

The divergence corresponds to the Poisson exponential family and is
commonly used for count data, topic models, and nonnegative matrix
factorization.

Classes
-------
GKLDivergence
    Generalized Kullback-Leibler divergence.

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

@BregmanDivergenceFactory.register("gkl")
class GKLDivergence(BaseBregmanDivergence):
    """
    Generalised Kullback-Leibler (I-divergence).

    Generator   φ(x)    = Σᵢ xᵢ ln(xᵢ)
    Divergence  D(x, y) = Σᵢ [ xᵢ ln(xᵢ/yᵢ) − (xᵢ − yᵢ) ]
    Gradient    ∇φ(x)   = ln(x) + 1

    Exponential family : Poisson
    Domain             : (0, +∞)ᵈ

    Note
    ----
    0 ln(0) = 0  and  0 ln(0/y) = 0  (continuity limit, paper §0/0 = 0).
    """

    name = "GKL"
    family = "Poisson"

    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.sum(np.where(X > 0, X * np.log(X), 0.0), axis=1)

    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(X, dtype=np.float64)) + 1.0
    
    def in_domain(self, X: np.ndarray) -> bool:
        return np.all(X > 0)
    
    def distance(self, X: np.ndarray, Y: np.ndarray, *, clip: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        logX = np.log(X)
        logY = np.log(Y)

        xlogx = np.sum(X * logX, axis=1)[:, None]
        xlogy = X @ logY.T

        xsum = np.sum(X, axis=1)[:, None]
        ysum = np.sum(Y, axis=1)[None, :]

        D = xlogx - xlogy - xsum + ysum
        
        if clip:
            np.maximum(D, 0.0, out=D)
        return D
