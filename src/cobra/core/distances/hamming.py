"""
Hamming distance implementation for COBRA framework.

This module provides an efficient computation of the Hamming distance
between two discrete or binary feature vectors.

Definition:

    d(x, y) = (1 / d) * sum_i [x_i != y_i]

where d is the number of features.

Hamming distance is primarily used for:
- categorical data comparison
- binary feature vectors
- discrete similarity estimation
- ensemble aggregation over symbolic representations
"""

from __future__ import annotations

import numpy as np
import numba as nb

from cobra.core.distances.base import BaseDistance
from cobra.core.distances.base import DistanceFactory

@nb.jit(nopython=True, parallel=True, fastmath=True)
def hamming_matrix_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Hamming distance using Numba acceleration.

    Parameters
    ----------
    x : np.ndarray
        First dataset of shape (n_samples_x, n_features).

    y : np.ndarray
        Second dataset of shape (n_samples_y, n_features).

    Returns
    -------
    np.ndarray
        Pairwise normalized Hamming distance matrix.
    """
    n_x, n_y = x.shape[0], y.shape[0]
    n_features = x.shape[1]

    distances = np.empty((n_x, n_y), dtype=np.float64)

    for i in nb.prange(n_x):
        for j in range(n_y):
            diff_count = 0.0
            for k in range(n_features):
                if x[i, k] != y[j, k]:
                    diff_count += 1.0

            distances[i, j] = diff_count / n_features

    return distances

@DistanceFactory.register("hamming")
class HammingDistance(BaseDistance):
    """
    Hamming distance metric.

    This class computes normalized Hamming distance between two datasets.
    It first attempts a Numba-accelerated implementation for speed,
    and falls back to a NumPy vectorized version if needed.

    Methods
    -------
    matrix(x, y)
        Compute pairwise Hamming distance matrix.

    Notes
    -----
    This metric assumes discrete or binary input features.
    Continuous features should be discretized before use.
    """

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Hamming distance matrix.

        Parameters
        ----------
        x : np.ndarray
            First dataset of shape (n_samples_x, n_features).

        y : np.ndarray
            Second dataset of shape (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Normalized Hamming distance matrix in range [0, 1].
        """
        x = np.asarray(x)
        y = np.asarray(y)

        try:
            return hamming_matrix_numba(x, y)
        except Exception:
            # fallback pure NumPy implementation
            return np.mean(
                x[:, None, :] != y[None, :, :],
                axis=2,
            )
