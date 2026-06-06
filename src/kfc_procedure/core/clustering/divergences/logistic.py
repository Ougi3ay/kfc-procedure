"""
Logistic Bregman divergence.

This module implements the logistic divergence induced by the generator

    phi(x)
    =
    Σ [x_i log(x_i)
       + (1 - x_i) log(1 - x_i)].

The resulting divergence corresponds to the binary cross-entropy loss

    D(x, y)
    =
    Σ [x_i log(x_i / y_i)
       + (1 - x_i)
         log((1 - x_i)/(1 - y_i))].

The divergence is associated with the Bernoulli exponential family and
is commonly used for binary observations and probabilistic clustering.

Classes
-------
LogisticLoss
    Logistic Bregman divergence.

Domain
------
All input values must satisfy

    0 < x_i < 1.

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

@BregmanDivergenceFactory.register("logistic")
class LogisticLoss(BaseBregmanDivergence):
    """
    Logistic / binary cross-entropy loss.

    Generator  φ(x) = Σᵢ [ xᵢ ln(xᵢ) + (1−xᵢ) ln(1−xᵢ) ]
    Divergence D(x, y) = Σᵢ [ xᵢ ln(xᵢ/yᵢ) + (1−xᵢ) ln((1−xᵢ)/(1−yᵢ)) ]

    Exponential family : Bernoulli / Binomial
    Domain             : (0, 1)ᵈ
    """

    name    = "Logit"
    family  = "Bernoulli / Binomial"


    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.sum(
            X * np.log(X) + (1 - X) * np.log(1 - X),
            axis=1
        )
    
    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.log(X) - np.log1p(-X)
    
    def in_domain(self, X: np.ndarray) -> bool:
        return np.all((0 < X) & (X < 1))

    def distance(self, X: np.ndarray, Y: np.ndarray, *, clip: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = np.clip(np.asarray(Y, dtype=np.float64), 1e-12, 1 - 1e-12)

        logY = np.log(Y)
        log1mY = np.log1p(-Y)

        term1 = X @ logY.T
        term2 = (1 - X) @ log1mY.T

        D = -term1 - term2 + self.phi(X)[:, None] + self.phi(Y)[None, :]

        if clip:
            np.maximum(D, 0.0, out=D)
        return D