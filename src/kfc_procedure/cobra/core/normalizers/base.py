"""
Normalization module for COBRA framework.

This module defines the abstract interface and factory system for all
normalization strategies used in the COBRA pipeline.

Normalization is a key transformation step applied to:
- input feature space
- estimator prediction space
- distance computation space

The goal is to ensure numerical stability and comparability across
different estimators and feature representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

import numpy as np

from cobra.core.factory import BaseFactory


class BaseNormalizer(ABC):
    """
    Abstract base class for all normalization strategies.

    This class defines a unified interface for transforming data
    into normalized representations suitable for downstream COBRA
    components such as distance functions and kernels.

    Methods
    -------
    fit(x)
        Compute normalization statistics from input data.

    transform(x)
        Apply normalization using learned statistics.

    fit_transform(x)
        Convenience method combining fit and transform.

    Notes
    -----
    All normalizers must be stateful and store learned statistics
    internally (e.g., mean, variance, min/max).
    """

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs) -> "BaseNormalizer":
        """
        Learn normalization parameters from input data.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        BaseNormalizer
            Fitted normalizer instance.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform input data using learned normalization parameters.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Normalized array with same shape as input.
        """
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Normalized data.
        """
        self.fit(x, **kwargs)
        return self.transform(x, **kwargs)


class NormalizerFactory(BaseFactory):
    """
    Factory class for normalization strategies.

    This factory enables dynamic registration and instantiation of
    normalizers using string identifiers.

    Examples
    --------
    >>> norm = NormalizerFactory.create("standard")
    >>> Xn = norm.fit_transform(X)
    """
    pass