"""
Euclidean distance implementation for COBRA framework.

This module provides a vectorized implementation of the L2 (Euclidean)
distance metric, which is widely used in machine learning for measuring
geometric similarity between feature vectors.

The Euclidean distance is defined as:

    d(x, y) = sqrt(||x - y||^2)

It is used in:
- kernel construction
- similarity-based aggregation
- clustering and nearest-neighbor estimation
"""

from __future__ import annotations

import numpy as np

from kfc_procedure.cobra.core.distances.base import BaseDistance
from kfc_procedure.cobra.core.distances.base import DistanceFactory


@DistanceFactory.register("euclidean", "l2")
class EuclideanDistance(BaseDistance):
    """
    Euclidean (L2) distance metric.

    This class computes the pairwise Euclidean distance between
    two datasets using a numerically stable vectorized formulation.

    Methods
    -------
    matrix(x, y)
        Compute full pairwise distance matrix.

    Notes
    -----
    The implementation uses the identity:

        ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>

    This avoids explicit loops and improves computational efficiency.
    """

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distance matrix.

        Parameters
        ----------
        x : np.ndarray
            First dataset of shape (n_samples_x, n_features).

        y : np.ndarray
            Second dataset of shape (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples_x, n_samples_y),
            where entry (i, j) is the Euclidean distance between
            x[i] and y[j].
        """
        x = np.asarray(x)
        y = np.asarray(y)

        x2 = np.sum(x ** 2, axis=1, keepdims=True)
        y2 = np.sum(y ** 2, axis=1, keepdims=True).T
        xy = x @ y.T

        dist = np.sqrt(np.maximum(x2 + y2 - 2 * xy, 0.0))
        return dist
