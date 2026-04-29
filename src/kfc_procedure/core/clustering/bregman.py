"""
Bregman K-Means clustering implementation
-----------------------------------------

Implements Lloyd-style clustering with arbitrary Bregman divergences.

Reference
---------
Has, Fischer & Mougeot (Algorithm 1)
Banerjee et al. (2005), JMLR
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.clustering.divergences.base import (
    BaseBregmanDivergence,
    BregmanDivergenceFactory,
)


# ============================================================
# Validation
# ============================================================

def validate_divergence_domain(div: BaseBregmanDivergence, X: np.ndarray) -> None:
    """
    Ensure input data is valid for the chosen divergence.

    Checks:
    - finite values only
    - domain constraints of divergence
    """
    X = np.asarray(X, dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            f"[{div.name}] Input contains NaN or Inf."
        )

    if not div.in_domain(X):
        raise ValueError(
            f"[{div.name}] Input outside valid domain."
        )


# ============================================================
# Main Model
# ============================================================

class BregmanKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    K-Means clustering with Bregman divergences.

    Uses Lloyd's algorithm:
        1. Assign points to closest centroid
        2. Update centroids using Bregman mean (Euclidean mean here)
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        divergence: Union[BaseBregmanDivergence, str] = "euclidean",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # --------------------------------------------------------
    # divergence
    # --------------------------------------------------------

    def _get_divergence(self) -> BaseBregmanDivergence:
        if isinstance(self.divergence, str):
            return BregmanDivergenceFactory.create(self.divergence)
        return self.divergence

    # --------------------------------------------------------
    # initialization
    # --------------------------------------------------------

    def _init_centroids(
        self,
        X: np.ndarray,
        rng,
        init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if init is not None:
            init = check_array(init, dtype=float, copy=True)
            if init.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("Invalid init shape")
            return init

        idx = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].copy()

    # --------------------------------------------------------
    # centroid update (Euclidean mean surrogate)
    # --------------------------------------------------------

    def _compute_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        K: int,
        rng,
    ) -> np.ndarray:
        """
        Compute cluster means.

        Handles empty clusters by reinitializing randomly.
        """
        n, d = X.shape
        centroids = np.zeros((K, d), dtype=X.dtype)

        counts = np.bincount(labels, minlength=K)

        np.add.at(centroids, labels, X)

        for k in range(K):
            if counts[k] == 0:
                centroids[k] = X[rng.randint(n)]
            else:
                centroids[k] /= counts[k]

        return centroids

    # --------------------------------------------------------
    # distortion
    # --------------------------------------------------------

    @staticmethod
    def _distortion(
        div: BaseBregmanDivergence,
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        D = div.distance(X, centroids)
        return float(np.mean(np.min(D, axis=1)))

    # streaming version (memory safe)
    def _distortion_stream(
        self,
        div: BaseBregmanDivergence,
        X: np.ndarray,
        centroids: np.ndarray,
        block: int = 4096,
    ) -> float:
        n = X.shape[0]
        total = 0.0

        for i in range(0, n, block):
            D = div.distance(X[i:i + block], centroids)
            total += np.sum(np.min(D, axis=1))

        return total / n

    # --------------------------------------------------------
    # Lloyd iteration
    # --------------------------------------------------------

    def _Lloyd(
        self,
        X: np.ndarray,
        div: BaseBregmanDivergence,
        rng,
        init: Optional[np.ndarray] = None,
    ):
        n, K = X.shape[0], self.n_clusters

        centroids = self._init_centroids(X, rng, init)
        prev = np.inf

        for it in range(self.max_iter):

            # E-step
            D = div.distance(X, centroids)
            labels = np.argmin(D, axis=1)

            # M-step
            centroids = self._compute_centroids(X, labels, K, rng)

            # distortion
            dist = self._distortion_stream(div, X, centroids)

            if self.verbose:
                print(f"iter={it} distortion={dist:.6f}")

            # convergence
            change = abs(prev - dist) / (abs(prev) + 1e-12)
            if change < self.tol:
                break

            prev = dist

        return labels, centroids, dist, it + 1

    # --------------------------------------------------------
    # fit
    # --------------------------------------------------------

    def fit(self, X: ArrayLike, y=None, init=None):

        X = check_array(X, dtype=float, ensure_2d=True)

        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples < n_clusters")

        rng = check_random_state(self.random_state)
        div = self._get_divergence()

        validate_divergence_domain(div, X)

        best = (None, None, np.inf, 0)

        n_init = 1 if init is not None else self.n_init

        for _ in range(n_init):
            labels, centroids, dist, it = self._Lloyd(X, div, rng, init)

            if dist < best[2]:
                best = (labels, centroids, dist, it)

        self.labels_ = best[0]
        self.cluster_centers_ = best[1]
        self.inertia_ = best[2]
        self.n_iter_ = best[3]
        self._divergence = div

        return self

    # --------------------------------------------------------
    # predict
    # --------------------------------------------------------

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return self._divergence.assign_clusters(X, self.cluster_centers_)

    # --------------------------------------------------------
    # transform
    # --------------------------------------------------------

    def transform(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return self._divergence.distance(X, self.cluster_centers_)

    # --------------------------------------------------------
    # sklearn API
    # --------------------------------------------------------

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def score(self, X, y=None) -> float:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return -self._distortion(self._divergence, X, self.cluster_centers_)

    # --------------------------------------------------------
    # repr
    # --------------------------------------------------------

    def __repr__(self):
        name = (
            self.divergence
            if isinstance(self.divergence, str)
            else type(self.divergence).__name__
        )
        return (
            f"BregmanKMeans("
            f"n_clusters={self.n_clusters}, "
            f"divergence={name}, "
            f"n_init={self.n_init})"
        )
