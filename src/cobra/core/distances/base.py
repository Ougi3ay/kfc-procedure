"""
Distance module for pairwise similarity computation in the COBRA pipeline.

This module defines the distance computation stage, where pairwise
distances between samples are calculated before kernel adaptation
and aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The distance stage measures similarity (or dissimilarity) between
samples in either:

- input space (feature space)
- prediction space (estimator output space)

These distance matrices are later transformed by the kernel adapter
and passed into kernel functions for neighbor selection and weighting.

Typical supported metrics include:

- Euclidean distance
- Manhattan distance
- Mahalanobis distance
- Minkowski distance
- custom task-specific metrics

By separating distance computation into dedicated modules, the
framework becomes:

- modular
- easily extensible
- optimization-friendly
- compatible with multiple COBRA variants

Examples
--------
>>> @DistanceFactory.register("euclidean")
... class EuclideanDistance(BaseDistance):
...     def matrix(self, x, y):
...         return np.linalg.norm(
...             x[:, None] - y[None, :],
...             axis=2
...         )

>>> distance = DistanceFactory.create("euclidean")
>>> D = distance.matrix(X_train, X_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cobra.core.factory import BaseFactory


class BaseDistance(ABC):
    """
    Abstract base class for distance computation strategies.

    Distance modules compute pairwise distance matrices between
    two input arrays.

    These distances are used as the foundation for:

    - kernel weighting
    - neighbor selection
    - optimization of aggregation models

    Parameters
    ----------
    **kwargs : dict
        Optional configuration parameters for the distance strategy.

    Attributes
    ----------
    params : dict
        Internal dictionary storing distance parameters.

    Notes
    -----
    Subclasses must implement the ``matrix()`` method.

    Registration and instantiation are typically handled using
    ``DistanceFactory``.

    Examples
    --------
    >>> class EuclideanDistance(BaseDistance):
    ...     def matrix(self, x, y):
    ...         return np.linalg.norm(
    ...             x[:, None] - y[None, :],
    ...             axis=2
    ...         )
    """

    def __init__(self, **kwargs):
        """
        Initialize distance object with optional parameters.

        Parameters
        ----------
        **kwargs : dict
            Distance configuration parameters.
        """
        self.params = dict(kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_params(self, **params):
        """
        Update distance parameters.

        Parameters
        ----------
        **params : dict
            Parameters to update.

        Returns
        -------
        BaseDistance
            Returns self for method chaining.

        Examples
        --------
        >>> distance.set_params(p=2)
        """
        for k, v in params.items():
            setattr(self, k, v)
            self.params[k] = v

        return self

    def get_params(self, deep=True):
        """
        Return stored distance parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility.
            Currently not used.

        Returns
        -------
        dict
            Dictionary of stored parameters.

        Examples
        --------
        >>> distance.get_params()
        {'p': 2}
        """
        return dict(self.params)

    @abstractmethod
    def matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the pairwise distance matrix between two arrays.

        Parameters
        ----------
        x : np.ndarray
            First input array of shape
            (n_samples_x, n_features).

        y : np.ndarray
            Second input array of shape
            (n_samples_y, n_features).

        Returns
        -------
        np.ndarray
            Distance matrix of shape
            (n_samples_x, n_samples_y).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> D = distance.matrix(X_train, X_test)
        """
        raise NotImplementedError


class DistanceFactory(BaseFactory):
    """
    Factory for ``BaseDistance`` implementations.

    This registry-based factory enables dynamic creation of
    distance strategies using string identifiers.

    It is especially useful for:

    - YAML-based pipeline configuration
    - model experimentation
    - hyperparameter optimization
    - benchmarking multiple distance metrics

    Examples
    --------
    >>> distance = DistanceFactory.create("euclidean")

    >>> D = distance.matrix(
    ...     X_train,
    ...     X_test
    ... )
    """
    pass
