"""
Min-Max normalization module.

This module scales features to a fixed range [0, 1] using:

    x' = (x - min) / (max - min)

This is commonly used when bounded input representation is required,
especially for distance-based learning systems like COBRA.
"""

import numpy as np

from cobra.core.normalizers.base import BaseNormalizer
from cobra.core.normalizers.base import NormalizerFactory


@NormalizerFactory.register("minmax")
class MinMaxNormalizer(BaseNormalizer):
    """
    Min-Max normalizer.

    Attributes
    ----------
    min_ : np.ndarray
        Minimum value per feature.

    max_ : np.ndarray
        Maximum value per feature.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray, **kwargs) -> "MinMaxNormalizer":
        """
        Compute feature-wise minimum and maximum.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        MinMaxNormalizer
            Fitted instance.
        """
        x = np.asarray(x)

        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        return self

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Scale features to [0, 1].

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
        return (x - self.min_) / (self.max_ - self.min_ + 1e-12)
