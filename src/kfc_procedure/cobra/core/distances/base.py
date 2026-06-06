"""
Distance module for COBRA framework.

This module defines the abstract interface for distance computation
between samples and provides a factory system for dynamic distance
metric registration.

Distance functions are a core component of COBRA, used in:
- kernel construction
- similarity computation
- aggregation weighting
- optimization objectives
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from kfc_procedure.cobra.core.factory import BaseFactory


class BaseDistance(ABC):
    """
    Abstract base class for all distance metrics.

    This class defines a unified interface for computing pairwise
    distance matrices between two sets of samples.

    Attributes
    ----------
    params : dict
        Dictionary storing all hyperparameters of the distance metric.

    Methods
    -------
    matrix(x, y)
        Compute pairwise distance matrix between x and y.

    get_params()
        Return stored parameters.

    set_params(**params)
        Update parameters dynamically.
    """

    def __init__(self, **kwargs):
        """
        Initialize distance metric with optional parameters.

        Parameters
        ----------
        **kwargs : dict
            Hyperparameters for the distance function.
        """
        self.params: Dict[str, Any] = dict(kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_params(self, **params):
        """
        Set parameters for the distance function.

        This method updates both internal attributes and parameter
        dictionary.

        Parameters
        ----------
        **params : dict
            Key-value pairs of parameters to update.

        Returns
        -------
        BaseDistance
            Updated instance (for chaining).
        """
        for k, v in params.items():
            setattr(self, k, v)
            self.params[k] = v

        return self

    def get_params(self, deep: bool = True):
        """
        Get parameters of the distance function.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility (not used).

        Returns
        -------
        dict
            Dictionary of parameters.
        """
        return dict(self.params)

    @abstractmethod
    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix between two datasets.

        Parameters
        ----------
        x : np.ndarray
            First dataset of shape (n_samples_x, n_features).

        y : np.ndarray
            Second dataset of shape (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples_x, n_samples_y).
        """
        raise NotImplementedError


class DistanceFactory(BaseFactory):
    """
    Factory class for distance metrics.

    This factory enables dynamic registration and instantiation of
    distance functions using string identifiers.

    Examples
    --------
    >>> dist = DistanceFactory.create("euclidean")
    >>> D = dist.matrix(X, Y)
    """
    pass
