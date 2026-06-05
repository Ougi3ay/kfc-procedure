"""
Manhattan (L1) distance implementation for COBRA framework.

This module provides a vectorized implementation of the Manhattan
distance, also known as L1 distance or city-block distance.

Definition:

    d(x, y) = sum_i |x_i - y_i|

The Manhattan distance is commonly used in:
- robust similarity estimation
- high-dimensional feature spaces
- sparse data settings
- kernel-based aggregation methods in COBRA
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance_matrix

from cobra.core.distances.base import BaseDistance
from cobra.core.distances.base import DistanceFactory


@DistanceFactory.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    """
    Manhattan (L1) distance metric.

    This class computes pairwise L1 distances between two datasets
    using SciPy's optimized distance computation utilities.

    Methods
    -------
    matrix(x, y)
        Compute pairwise Manhattan distance matrix.

    Notes
    -----
    L1 distance is more robust than L2 in high-dimensional or
    noisy feature spaces, as it reduces sensitivity to outliers.
    """

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Manhattan distance matrix.

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
            where entry (i, j) represents the L1 distance between
            x[i] and y[j].
        """
        x = np.asarray(x)
        y = np.asarray(y)

        return distance_matrix(x, y, p=1)
