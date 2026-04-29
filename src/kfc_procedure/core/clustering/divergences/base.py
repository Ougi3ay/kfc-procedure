"""
kfc_procedure.divergences.base
-------------------------------
BregmanDivergence abstract base class and BregmanDivergenceFactory registry.

A Bregman divergence is fully defined by a strictly convex generator φ.
Subclasses only need to implement ``phi`` and ``grad_phi``; the vectorised
pairwise ``distance`` method is provided here for free.

Concrete divergences  (Table 1 of Has, Fischer & Mougeot)
---------------------------------------------------------
  SquaredEuclidean       – Euclid  – Gaussian family      – domain ℝᵈ
  GeneralKullbackLeibler – GKL     – Poisson  family      – domain (0,+∞)ᵈ
  LogisticDivergence     – Logit   – Bernoulli/Bin family – domain (0,1)ᵈ
  ItakuraSaito           – Ita     – Exponential family   – domain (0,+∞)ᵈ

Reference
---------
Banerjee et al. (2005). "Clustering with Bregman Divergences." JMLR 6,
1705–1749.  Equation (1) defines D_φ(x, y) = φ(x) − φ(y) − ⟨x−y, ∇φ(y)⟩.
Has, Fischer & Mougeot. "A clusterwise supervised learning procedure based
    on aggregation of distances."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Type
import numpy as np

from kfc_procedure.core.factory import BaseFactory


class BaseBregmanDivergence(ABC):
    """
    Abstract Bregman divergence.

    Subclasses must implement:
        ``domain``      – valid input domain (e.g. ℝᵈ, (0, +∞)ᵈ, etc.)
        ``phi(X)``      – generator function,  shape → (n,)
        ``grad_phi(X)`` – gradient of phi,     shape → (n, d)

    The ``distance`` method is provided and fully vectorised:
        D_φ(x, y) = φ(x) − φ(y) − ⟨∇φ(y), x − y⟩
        D_φ(x, y) = φ(x) − φ(y) − ⟨∇φ(y), x⟩ + ⟨∇φ(y), y⟩
    returning an (n_samples, n_centroids) matrix.

    The ``centroid`` (mean as minimiser property, Banerjee et al. 2005) is
    simply the arithmetic mean for all Bregman divergences, so ``centroid``
    is provided here as a final method.
    """

    name: ClassVar[str] = "base"
    family: ClassVar[str] = "base"

    def __init__(self, validate_domain: bool = True, **kwargs):
        self.validate_domain = validate_domain
        self._cache_key = None
        self._phi_Y     = None
        self._grad_Y    = None
        self._dot_Y     = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def in_domain(self, X: np.ndarray) -> bool:
        """Check if X is in the valid domain for this divergence."""

    @abstractmethod
    def phi(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the generator φ(X) row-wise.

        Parameters
        ----------
            X : ndarray, shape (n_samples, d)
                Points inside the domain C of φ.
        
        Returns
        -------
            ndarray, shape (n,)
                φ(X[i]) for each row i.
        """

    @abstractmethod
    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate ∇φ row-wise.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Points inside the interior of the domain C.

        Returns
        -------
        ndarray, shape (n, d)
            ∇φ(X[i]) for each row i.
        """

    def distance(self, X: np.ndarray, Y: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """
        Vectorised pairwise Bregman divergence D_φ(X[i], Y[k]).
        
        Formula  : D_φ(x, y) = φ(x) − φ(y) − ⟨∇φ(y), x − y⟩
        Optimize : D_φ(x, y) = φ(x)[:, None] - φ(y)[None, :] - (X @ ∇φ(y).T - diag(∇φ(y) @ y.T))

        Parameters
        ----------
        X : (n_samples,   d)
        Y : (n_centroids, d)

        Returns
        -------
        D : (n_samples, n_centroids)   D[i, k] = D_φ(X[i], Y[k])
        """
        X = np.asarray(X, dtype=np.float64, order="C", copy=False)
        Y = np.asarray(Y, dtype=np.float64, order="C", copy=False)
        
        if not (self.in_domain(X) and self.in_domain(Y)):
            raise ValueError(f"X/Y outside domain of {self.name}")
        
        y_key = (Y.__array_interface__["data"][0], Y.shape, Y.strides, Y.dtype)
        if self._cache_key != y_key:
            self._phi_Y = self.phi(Y)
            self._grad_Y= np.asarray(self.grad_phi(Y), dtype=np.float64, order="F")
            self._dot_Y = np.einsum("kd,kd->k", self._grad_Y, Y) # precompute ∇φ(Y) @ Y.T diagonal for all centroids
            self._cache_key = y_key
        
        # Compute D (n, K)
        phi_X = self.phi(X)[:, None]
        XG    = X @ self._grad_Y.T
        D     = (
                phi_X
                - self._phi_Y[None, :]
                - XG
                + self._dot_Y[None, :]
        )

        if clip:
            return np.maximum(D, 0.0, out=D)
        return D


    # Derived helpers – all delegate to distance(), no duplication
    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Alias for ``distance``.  Returns the (n, K) divergence matrix.

        Kept for backward compatibility.  Always delegates to the
        fully vectorised ``distance`` method – never loops in Python.
        """
        return self.distance(X, Y)

    def centroid(self, X: np.ndarray) -> np.ndarray:
        """
        Return the cluster centroid of the point cloud X.

        By Proposition 1 (Banerjee et al. 2005a – mean-as-minimiser), the
        arithmetic mean minimises E[D_φ(X, c)] for *every* Bregman divergence,
        so this implementation is correct regardless of which subclass is used.

        Parameters
        ----------
        X : ndarray, shape (n, d)

        Returns
        -------
        ndarray, shape (d,)
        """

        return np.mean(np.asarray(X, dtype=float), axis=0)

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point in X to the nearest centroid.

        Uses the vectorised ``distance`` method: one matrix call, no loops.

        Parameters
        ----------
        X         : ndarray, shape (n, d)
        centroids : ndarray, shape (K, d)

        Returns
        -------
        labels : ndarray of int, shape (n,)
            labels[i] = argmin_k D_φ(X[i], centroids[k])
        """
        D = self.distance(X, centroids)
        return np.argmin(D, axis=1)

    def __repr__(self) -> str:
        attrs = {k: getattr(self, k) for k in self.__dict__ if not k.startswith("_")}
        return f"{self.__class__.__name__} {attrs})"


class BregmanDivergenceFactory(BaseFactory):
    """
    Registry of named BregmanDivergence subclasses.
    """

    _registry: Dict[str, Type[BaseBregmanDivergence]] = {}