"""
Bregman K-Means clustering.

This module implements the Bregman K-Means algorithm, a generalization
of classical k-means in which the Euclidean distance is replaced by an
arbitrary Bregman divergence.

The algorithm follows Lloyd's iterative optimization procedure:

1. Assign each sample to the nearest centroid according to a chosen
   Bregman divergence.
2. Update cluster centroids using the arithmetic mean of assigned
   samples.
3. Repeat until convergence or until the maximum number of iterations
   is reached.

The implementation supports any divergence derived from
``BaseBregmanDivergence`` and is compatible with the scikit-learn
estimator API.

Functions
---------
validate_divergence_domain
    Validate that input data satisfy the domain requirements of a
    divergence.

Classes
-------
BregmanKMeans
    Lloyd-style clustering estimator based on Bregman divergences.

Notes
-----
For a strictly convex generator function ``phi``, the Bregman
divergence between points ``x`` and ``y`` is

    D_phi(x, y)
    =
    phi(x)
    - phi(y)
    - <grad_phi(y), x - y>

Common special cases include:

* Squared Euclidean divergence (Gaussian family)
* Generalized Kullback-Leibler divergence (Poisson family)
* Itakura-Saito divergence (Gamma family)
* Logistic divergence (Bernoulli family)

This implementation uses multiple random initializations and selects
the solution with the lowest average cluster distortion.

References
----------
Banerjee, A., Merugu, S., Dhillon, I. S., and Ghosh, J. (2005).
"Clustering with Bregman Divergences."
Journal of Machine Learning Research, 6, 1705-1749.

Lloyd, S. P. (1982).
"Least Squares Quantization in PCM."
IEEE Transactions on Information Theory, 28(2), 129-137.

Hartigan, J. A., and Wong, M. A. (1979).
"A K-Means Clustering Algorithm."
Applied Statistics, 28(1), 100-108.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.clustering.divergences.base import (
    BaseBregmanDivergence
)


def validate_divergence_domain(div: BaseBregmanDivergence, X: np.ndarray) -> None:
    """
    Ensure input data is valid for the chosen divergence.

    Parameters
    ----------
    div : BaseBregmanDivergence
        Divergence instance with domain constraints
    X : np.ndarray
        Input data to validate

    Raises
    ------
    ValueError
        If X contains NaN/Inf or violates divergence domain constraints.

    Checks:
    - finite values only (no NaN or Inf)
    - domain constraints of divergence (e.g., positivity for log-based divergences)
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


class BregmanKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    K-Means clustering with Bregman divergences.

    Uses Lloyd's algorithm with pluggable Bregman divergences:
        1. Assign points to closest centroid (using Bregman distance)
        2. Update centroids using Euclidean mean surrogate
        3. Repeat until convergence

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters
    divergence : BaseBregmanDivergence
        Divergence metric to use for distance computation
    n_init : int, default=10
        Number of random initializations
    max_iter : int, default=300
        Maximum iterations per initialization
    tol : float, default=1e-4
        Convergence tolerance on relative distortion change
    random_state : int or RandomState, default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Enable iteration logging

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment for each sample
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centroids
    inertia_ : float
        Sum of squared distances to nearest cluster center
    n_iter_ : int
        Number of iterations run for best initialization
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        divergence: BaseBregmanDivergence,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
        verbose: bool = False,
    ):
        if not isinstance(divergence, BaseBregmanDivergence):
            raise TypeError(
                "divergence must be an instance of BaseBregmanDivergence"
            )
        
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def _init_centroids(
        self,
        X: np.ndarray,
        rng,
        init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize cluster centroids.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        rng : RandomState
            Random state for reproducible initialization
        init : ndarray of shape (n_clusters, n_features), optional
            Custom initial centroids. If provided, must have correct shape.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Initial cluster centroids
        """
        if init is not None:
            init = check_array(init, dtype=float, copy=True)
            # Validate initialization shape matches expected dimensions
            if init.shape != (self.n_clusters, X.shape[1]):
                raise ValueError(
                    f"init shape {init.shape} does not match "
                    f"(n_clusters, n_features)=({self.n_clusters}, {X.shape[1]})"
                )
            return init

        # Random k-means++ style initialization: sample random data points
        idx = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].copy()

    def _compute_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        K: int,
        rng,
    ) -> np.ndarray:
        """
        Compute cluster means (M-step of Lloyd's algorithm).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        labels : ndarray of shape (n_samples,)
            Current cluster assignments
        K : int
            Number of clusters
        rng : RandomState
            Random state for handling empty clusters

        Returns
        -------
        centroids : ndarray of shape (K, n_features)
            Updated cluster centroids (Euclidean means)

        Notes
        -----
        Empty clusters are reinitialized with a random data point.
        This prevents numerical issues from division by zero.
        """
        n, d = X.shape
        centroids = np.zeros((K, d), dtype=X.dtype)

        counts = np.bincount(labels, minlength=K)

        np.add.at(centroids, labels, X)

        for k in range(K):
            if counts[k] == 0:
                # ERROR HANDLING: Empty cluster detected - reinitialize randomly
                # This can occur if a cluster has no assigned points
                centroids[k] = X[rng.randint(n)]
            else:
                centroids[k] /= counts[k]

        return centroids

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

    def _Lloyd(
        self,
        X: np.ndarray,
        div: BaseBregmanDivergence,
        rng,
        init: Optional[np.ndarray] = None,
    ):
        """
        Execute one run of Lloyd's clustering algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        div : BaseBregmanDivergence
            Divergence metric for distance computation
        rng : RandomState
            Random state for reproducibility
        init : ndarray, optional
            Initial centroids

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster assignments
        centroids : ndarray of shape (n_clusters, n_features)
            Final centroids
        dist : float
            Final distortion (average minimum distance)
        n_iter : int
            Number of iterations until convergence
        """
        n, K = X.shape[0], self.n_clusters

        centroids = self._init_centroids(X, rng, init)
        prev = np.inf

        for it in range(self.max_iter):

            # E-step: Assign points to nearest centroid
            D = div.distance(X, centroids)
            labels = np.argmin(D, axis=1)

            # M-step: Update centroids
            centroids = self._compute_centroids(X, labels, K, rng)

            # Compute distortion (average minimum distance to centroid)
            dist = self._distortion_stream(div, X, centroids)

            if self.verbose:
                print(f"iter={it} distortion={dist:.6f}")

            # Convergence check: relative change in distortion
            change = abs(prev - dist) / (abs(prev) + 1e-12)
            if change < self.tol:
                break

            prev = dist

        return labels, centroids, dist, it + 1

    def fit(self, X: ArrayLike, y=None, init=None):

        X = check_array(X, dtype=float, ensure_2d=True)

        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples < n_clusters")

        rng = check_random_state(self.random_state)

        validate_divergence_domain(self.divergence, X)

        best = (None, None, np.inf, 0)

        n_init = 1 if init is not None else self.n_init

        for _ in range(n_init):
            labels, centroids, dist, it = self._Lloyd(X, self.divergence, rng, init)

            if dist < best[2]:
                best = (labels, centroids, dist, it)

        self.labels_ = best[0]
        self.cluster_centers_ = best[1]
        self.inertia_ = best[2]
        self.n_iter_ = best[3]

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return self.divergence.assign_clusters(X, self.cluster_centers_)

    def transform(self, X: ArrayLike) -> NDArray:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return self.divergence.distance(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def score(self, X, y=None) -> float:
        check_is_fitted(self)
        X = check_array(X, dtype=float, ensure_2d=True)
        return -self._distortion(self.divergence, X, self.cluster_centers_)

    def __repr__(self):
        return (
            f"BregmanKMeans("
            f"n_clusters={self.n_clusters}, "
            f"divergence={type(self.divergence).__name__}, "
            f"n_init={self.n_init})"
        )
