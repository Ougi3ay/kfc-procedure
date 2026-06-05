"""
Standard normalization (Z-score scaling).

This module implements standard score normalization, which transforms
data to have zero mean and unit variance:

    x' = (x - mean) / std

This is widely used in machine learning pipelines to stabilize
optimization and improve distance-based methods.
"""

import numpy as np

from cobra.core.normalizers.base import BaseNormalizer
from cobra.core.normalizers.base import NormalizerFactory


@NormalizerFactory.register("standard", "zscore")
class StandardNormalizer(BaseNormalizer):
    """
    Standard (Z-score) normalizer.

    This normalizer rescales each feature to have:
    - mean = 0
    - standard deviation = 1

    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature computed during fitting.

    std_ : np.ndarray
        Standard deviation of each feature computed during fitting.

    Notes
    -----
    A small epsilon is added to std for numerical stability.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray, **kwargs) -> "StandardNormalizer":
        """
        Compute mean and standard deviation.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        StandardNormalizer
            Fitted instance.
        """
        x = np.asarray(x)

        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-12

        return self

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Z-score normalization.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Normalized data.
        """
        x = np.asarray(x)
        return (x - self.mean_) / self.std_
