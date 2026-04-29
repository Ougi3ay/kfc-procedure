"""
Built-in distance implementations for the COBRA pipeline.

This module provides concrete implementations of ``BaseDistance`` used
in the distance computation stage of the COBRA architecture.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Implemented distances
---------------------

1. EuclideanDistance

    Standard L2 distance between samples.

    Logic:
        d(x, y) = sqrt(sum((x - y)^2))

2. ManhattanDistance

    L1 distance using absolute coordinate differences.

    Logic:
        d(x, y) = sum(|x - y|)

3. HammingDistance

    Fraction of coordinates that differ.

    Logic:
        d(x, y) = mean(x != y)

    Commonly used for categorical or binary features.

4. MinkowskiDistance

    Generalized Lp distance with configurable exponent ``p``.

    Logic:
        d(x, y) = (sum(|x - y|^p))^(1/p)

5. CosineDistance

    Distance based on cosine similarity.

    Logic:
        d(x, y) = 1 - cosine_similarity(x, y)

These implementations are automatically registered using
``DistanceFactory.register()`` and can be instantiated dynamically.

Examples
--------
>>> distance = DistanceFactory.create("euclidean")
>>> D = distance.matrix(X_train, X_test)

>>> distance = DistanceFactory.create("minkowski", p=4)
>>> D = distance.matrix(X_train, X_test)
"""

from __future__ import annotations

import numpy as np

from cobra.core.distances.base import (
    BaseDistance,
    DistanceFactory,
)


@DistanceFactory.register("euclidean", "l2")
class EuclideanDistance(BaseDistance):
    """
    Euclidean (L2) distance.

    This distance computes the standard straight-line distance
    between two samples in feature space.

    Mathematical form
    -----------------
    d(x, y) = sqrt(sum((x - y)^2))

    Notes
    -----
    This implementation uses a vectorized formulation for efficient
    large-scale pairwise computation.

    Examples
    --------
    >>> distance = EuclideanDistance()
    >>> D = distance.matrix(X_train, X_test)
    """

    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Euclidean distance matrix.

        Parameters
        ----------
        x : np.ndarray
            First input array.

        y : np.ndarray
            Second input array.

        Returns
        -------
        np.ndarray
            Pairwise Euclidean distance matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        x2 = np.sum(x ** 2, axis=1, keepdims=True)
        y2 = np.sum(y ** 2, axis=1, keepdims=True).T
        xy = x @ y.T

        return np.sqrt(
            np.maximum(x2 + y2 - 2 * xy, 0.0)
        )


@DistanceFactory.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    """
    Manhattan (L1) distance.

    This distance computes the sum of absolute differences
    across all feature dimensions.

    Mathematical form
    -----------------
    d(x, y) = sum(|x - y|)

    Examples
    --------
    >>> distance = ManhattanDistance()
    >>> D = distance.matrix(X_train, X_test)
    """

    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Manhattan distance matrix.

        Returns
        -------
        np.ndarray
            Pairwise Manhattan distance matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        return np.sum(
            np.abs(x[:, None, :] - y[None, :, :]),
            axis=2,
        )


@DistanceFactory.register("hamming")
class HammingDistance(BaseDistance):
    """
    Hamming distance.

    This distance computes the proportion of differing
    coordinates between samples.

    Mathematical form
    -----------------
    d(x, y) = mean(x != y)

    Commonly used for:

    - binary features
    - categorical variables
    - encoded label vectors

    Examples
    --------
    >>> distance = HammingDistance()
    >>> D = distance.matrix(X_train, X_test)
    """

    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Hamming distance matrix.

        Returns
        -------
        np.ndarray
            Pairwise Hamming distance matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        return np.mean(
            x[:, None, :] != y[None, :, :],
            axis=2,
        )


@DistanceFactory.register("minkowski", "lp")
class MinkowskiDistance(BaseDistance):
    """
    Minkowski (Lp) distance.

    This distance generalizes Euclidean and Manhattan distance
    using a configurable exponent ``p``.

    Special cases:

    - p = 1 → Manhattan distance
    - p = 2 → Euclidean distance

    Mathematical form
    -----------------
    d(x, y) = (sum(|x - y|^p))^(1/p)

    Parameters
    ----------
    p : float, default=3
        Exponent controlling the distance geometry.

    Examples
    --------
    >>> distance = MinkowskiDistance(p=4)
    >>> D = distance.matrix(X_train, X_test)
    """

    def __init__(
        self,
        p: float = 3,
        **kwargs,
    ):
        """
        Initialize Minkowski distance.

        Parameters
        ----------
        p : float, default=3
            Distance exponent.
        """
        super().__init__(p=p, **kwargs)

    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Minkowski distance matrix.

        Returns
        -------
        np.ndarray
            Pairwise Minkowski distance matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        p = self.p

        return np.sum(
            np.abs(x[:, None, :] - y[None, :, :]) ** p,
            axis=2,
        ) ** (1 / p)


@DistanceFactory.register("cosine")
class CosineDistance(BaseDistance):
    """
    Cosine distance.

    This distance measures angular dissimilarity between vectors
    based on cosine similarity.

    Mathematical form
    -----------------
    d(x, y) = 1 - cosine_similarity(x, y)

    Notes
    -----
    Useful when vector magnitude is less important than direction.

    Common applications include:

    - text embeddings
    - recommendation systems
    - high-dimensional sparse vectors

    Examples
    --------
    >>> distance = CosineDistance()
    >>> D = distance.matrix(X_train, X_test)
    """

    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine distance matrix.

        Returns
        -------
        np.ndarray
            Pairwise cosine distance matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        x_norm = np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        )

        y_norm = np.linalg.norm(
            y,
            axis=1,
            keepdims=True,
        ).T

        sim = (
            x @ y.T
        ) / (x_norm * y_norm + 1e-12)

        return 1.0 - sim
