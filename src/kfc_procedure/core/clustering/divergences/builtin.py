"""
kfc_procedure.divergences.builtin
----------------------------------
The four Bregman divergences used in the KFC paper (Has et al., Table 1).

Each class self-registers with BregmanDivergenceFactory on import.

    Name          Generator φ(x)                    Model
    ----------    -----------------------------     ---------------
    "euclidean"   Σ xᵢ²                             Gaussian
    "gkl"         Σ xᵢ ln(xᵢ) − xᵢ                  Poisson
    "is"          −Σ ln(xᵢ)                         Gamma / spectral
    "logistic"    Σ xᵢ ln(xᵢ) + (1−xᵢ) ln(1−xᵢ)     Bernoulli
"""
from __future__ import annotations

import numpy as np

from kfc_procedure.core.clustering.divergences.base import BregmanDivergence, BregmanDivergenceFactory


@BregmanDivergenceFactory.register("euclidean")
class SquaredEuclidean(BregmanDivergence):
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


@BregmanDivergenceFactory.register("gkl")
class GKLDivergence(BregmanDivergence):
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


@BregmanDivergenceFactory.register("is")
class ItakuraSaito(BregmanDivergence):
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


@BregmanDivergenceFactory.register("logistic")
class LogisticLoss(BregmanDivergence):
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
