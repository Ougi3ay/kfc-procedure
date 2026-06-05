"""
Minkowski distance implementation for COBRA framework.

This module provides a generalization of several common distance
metrics including Euclidean (L2) and Manhattan (L1) distances.

Definition:

    d(x, y) = (sum_i |x_i - y_i|^p)^(1/p)

Special cases:
- p = 1 -> Manhattan distance (L1)
- p = 2 -> Euclidean distance (L2)

The Minkowski distance is widely used in:
- distance-based learning algorithms
- kernel methods
- similarity aggregation in ensemble systems (COBRA)
"""

from __future__ import annotations

import numpy as np

from cobra.core.distances.base import BaseDistance
from cobra.core.distances.base import DistanceFactory


# =========================================================
# Minkowski Distance
# =========================================================
@DistanceFactory.register("minkowski", "lp")
class MinkowskiDistance(BaseDistance):
    """
    Minkowski (Lp) distance metric.

    This class implements the generalized Lp norm distance between
    two datasets, parameterized by exponent p.

    Attributes
    ----------
    p : float
        Order of the norm (controls shape of the distance metric).

    Methods
    -------
    matrix(x, y)
        Compute pairwise Minkowski distance matrix.

    Notes
    -----
    - p = 1 -> Manhattan distance
    - p = 2 -> Euclidean distance
    - p -> ∞ approximates Chebyshev distance (not implemented here)
    """

    def __init__(self, p: float = 3, **kwargs):
        """
        Initialize Minkowski distance.

        Parameters
        ----------
        p : float, default=3
            Order of the Minkowski norm.
        **kwargs : dict
            Additional optional parameters (stored in BaseDistance).
        """
        super().__init__(p=p, **kwargs)

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Minkowski distance matrix.

        Parameters
        ----------
        x : np.ndarray
            First dataset of shape (n_samples_x, n_features).

        y : np.ndarray
            Second dataset of shape (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Pairwise Minkowski distance matrix of shape
            (n_samples_x, n_samples_y).
        """
        x = np.asarray(x)
        y = np.asarray(y)

        p = self.p

        return np.sum(
            np.abs(x[:, None, :] - y[None, :, :]) ** p,
            axis=2,
        ) ** (1 / p)
