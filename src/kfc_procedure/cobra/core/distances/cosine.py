"""
Cosine distance implementation for COBRA framework.

This module provides a vectorized implementation of cosine distance,
which measures angular dissimilarity between vectors.

Definition:

    cosine_similarity(x, y) = (x · y) / (||x|| * ||y||)

    cosine_distance(x, y) = 1 - cosine_similarity(x, y)

Cosine distance is widely used in:
- high-dimensional sparse data
- text embeddings
- similarity-based ensemble learning (COBRA)
- kernel construction
"""

from __future__ import annotations

import numpy as np

from kfc_procedure.cobra.core.distances.base import BaseDistance
from kfc_procedure.cobra.core.distances.base import DistanceFactory


# =========================================================
# Cosine Distance
# =========================================================
@DistanceFactory.register("cosine")
class CosineDistance(BaseDistance):
    """
    Cosine distance metric.

    This class computes pairwise cosine distance between two datasets
    using a numerically stable vectorized formulation.

    Methods
    -------
    matrix(x, y)
        Compute pairwise cosine distance matrix.

    Notes
    -----
    - Output range is [0, 2] in theory for cosine distance,
      but typically [0, 1] for non-negative data.
    - A small epsilon is used to avoid division by zero.
    """

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distance matrix.

        Parameters
        ----------
        x : np.ndarray
            First dataset of shape (n_samples_x, n_features).

        y : np.ndarray
            Second dataset of shape (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Cosine distance matrix of shape (n_samples_x, n_samples_y),
            where each entry represents angular dissimilarity.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True).T

        sim = (x @ y.T) / (x_norm * y_norm + 1e-12)

        return 1.0 - sim
