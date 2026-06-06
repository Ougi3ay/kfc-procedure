"""
Base interfaces for Bregman divergences.

This module defines the abstract foundation for all Bregman divergence
implementations used throughout the KFCProcedure clustering framework.

The central component is :class:`BaseBregmanDivergence`, which specifies
the common API required by divergence-based clustering algorithms. Each
concrete divergence must implement a convex generator function, its
gradient, and domain validation logic.

The module also provides :class:`BregmanDivergenceFactory`, a registry-
based factory responsible for discovering and instantiating divergence
implementations by name.

Classes
-------
BaseBregmanDivergence
    Abstract interface for Bregman divergence functions.

BregmanDivergenceFactory
    Factory and registry for divergence implementations.

Notes
-----
Bregman divergences generalize many important distance and divergence
measures used in machine learning, including:

* Squared Euclidean divergence
* Kullback-Leibler divergence
* Generalized I-divergence
* Itakura-Saito divergence
* Mahalanobis-type divergences

These divergences form the mathematical foundation of Bregman clustering
algorithms and centroid-based optimization procedures.

References
----------
Bregman, L. M. (1967).
"The relaxation method of finding the common point of convex sets and
its application to the solution of problems in convex programming."

Banerjee, A., Merugu, S., Dhillon, I. S., and Ghosh, J. (2005).
"Clustering with Bregman Divergences."
Journal of Machine Learning Research, 6, 1705-1749.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Type
import numpy as np

from kfc_procedure.core.factory import BaseFactory

class BaseBregmanDivergence(ABC):
    """
    Abstract base class for Bregman divergences.

    A Bregman divergence is a generalized distance measure induced by a
    strictly convex and differentiable generator function ``phi``.

    For points ``x`` and ``y``,

        D_phi(x, y) = phi(x) - phi(y) - <grad_phi(y), x - y>

    Unlike metric distances, Bregman divergences are generally not
    symmetric and do not necessarily satisfy the triangle inequality.
    They provide a unifying framework for many important divergence
    measures, including squared Euclidean distance, Kullback-Leibler
    divergence, Itakura-Saito divergence, and Mahalanobis-type
    divergences.

    This interface defines the common contract for implementing Bregman
    divergence families used in clustering, prototype learning, and
    centroid-based optimization algorithms. Concrete subclasses must
    provide the generator function, its gradient, and domain validation
    logic.

    Parameters
    ----------
    validate_domain : bool, default=True
        Whether to validate that input data belong to the valid domain
        of the divergence before computing distances.

    **kwargs : dict
        Additional divergence-specific parameters stored as instance
        attributes.

    Attributes
    ----------
    validate_domain : bool
        Whether domain validation is enabled.

    name : str
        Human-readable identifier of the divergence.

    family : str
        Name of the divergence family.

    Notes
    -----
    Subclasses must implement:

    * ``in_domain(X)``
    * ``phi(X)``
    * ``grad_phi(X)``

    The :meth:`distance` method caches quantities associated with the
    reference points ``Y`` to accelerate repeated divergence evaluations
    during iterative clustering procedures such as KFCProcedure and
    related Bregman clustering algorithms.

    References
    ----------
    Bregman, L. M. (1967).
    "The relaxation method of finding the common point of convex sets
    and its application to the solution of problems in convex
    programming."

    Banerjee, A., Merugu, S., Dhillon, I. S., and Ghosh, J. (2005).
    "Clustering with Bregman Divergences."
    Journal of Machine Learning Research, 6, 1705-1749.
    """

    name: ClassVar[str] = "base"
    family: ClassVar[str] = "base"

    def __init__(self, validate_domain: bool = True, **kwargs):
        """
        Initialize the divergence.

        Parameters
        ----------
        validate_domain : bool, default=True
            Whether to enforce domain validation before divergence
            computations.

        **kwargs : dict
            Additional divergence-specific configuration parameters.
            Each key-value pair is stored as an instance attribute.
        """
        self.validate_domain = validate_domain
        self._cache_key = None
        self._phi_Y = None
        self._grad_Y = None
        self._dot_Y = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def in_domain(self, X: np.ndarray) -> bool:
        """
        Check whether samples belong to the valid divergence domain.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        bool
            True if all samples satisfy the domain constraints,
            otherwise False.
        """
        pass

    @abstractmethod
    def phi(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the convex generator function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Generator function values evaluated at each sample.
        """
        pass

    @abstractmethod
    def grad_phi(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the generator function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Gradient of the generator function evaluated at each sample.
        """
        pass

    def distance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        """
        Compute pairwise Bregman divergences.

        Given a set of samples ``X`` and reference points ``Y``, this
        method computes the divergence matrix

        .. math::

            D_{ij}
            =
            D_{\\phi}(X_i, Y_j)

        for all sample-centroid pairs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Y : ndarray of shape (n_centroids, n_features)
            Reference points or centroids.

        clip : bool, default=True
            Whether to clip small negative numerical artifacts to zero.

        Returns
        -------
        ndarray of shape (n_samples, n_centroids)
            Pairwise divergence matrix where entry ``(i, j)``
            corresponds to ``D_phi(X[i], Y[j])``.

        Raises
        ------
        ValueError
            If either ``X`` or ``Y`` contains values outside the
            divergence domain.

        Notes
        -----
        Quantities dependent only on ``Y`` are cached internally
        to improve performance when repeatedly evaluating distances
        against the same reference points.
        """
        X = np.asarray(X, dtype=np.float64, order="C", copy=False)
        Y = np.asarray(Y, dtype=np.float64, order="C", copy=False)

        if not (self.in_domain(X) and self.in_domain(Y)):
            raise ValueError(f"X/Y outside domain of {self.name}")

        y_key = (
            Y.__array_interface__["data"][0],
            Y.shape,
            Y.strides,
            Y.dtype,
        )

        if self._cache_key != y_key:
            self._phi_Y = self.phi(Y)
            self._grad_Y = np.asarray(
                self.grad_phi(Y),
                dtype=np.float64,
                order="F",
            )
            self._dot_Y = np.einsum(
                "kd,kd->k",
                self._grad_Y,
                Y,
            )
            self._cache_key = y_key

        phi_X = self.phi(X)[:, None]
        XG = X @ self._grad_Y.T

        D = (
            phi_X
            - self._phi_Y[None, :]
            - XG
            + self._dot_Y[None, :]
        )

        if clip:
            return np.maximum(D, 0.0, out=D)

        return D

    def pairwise(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise divergences.

        This method is an alias for :meth:`distance`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Y : ndarray of shape (n_centroids, n_features)
            Reference points.

        Returns
        -------
        ndarray of shape (n_samples, n_centroids)
            Pairwise divergence matrix.
        """
        return self.distance(X, Y)

    def centroid(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the arithmetic centroid of a sample set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_features,)
            Arithmetic mean of the input samples.
        """
        return np.mean(np.asarray(X, dtype=float), axis=0)

    def assign_clusters(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """
        Assign samples to the nearest centroid.

        Each sample is assigned to the centroid that minimizes the
        corresponding Bregman divergence.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        centroids : ndarray of shape (n_clusters, n_features)
            Cluster centroids.

        Returns
        -------
        ndarray of shape (n_samples,)
            Index of the nearest centroid for each sample.
        """
        D = self.distance(X, centroids)
        return np.argmin(D, axis=1)

    def __repr__(self) -> str:
        """
        Return a string representation of the divergence.

        Returns
        -------
        str
            Representation containing the class name and public
            configuration attributes.
        """
        attrs = {
            k: getattr(self, k)
            for k in self.__dict__
            if not k.startswith("_")
        }
        return f"{self.__class__.__name__}({attrs})"


class BregmanDivergenceFactory(BaseFactory):
    """
    Factory class for Bregman divergence implementations.

    This factory maintains a registry mapping divergence identifiers
    to concrete subclasses of :class:`BaseBregmanDivergence`. It enables
    dynamic creation of divergence objects from configuration files,
    user parameters, or runtime specifications.

    Notes
    -----
    The registry associates string names with divergence classes.
    New divergences can be registered and instantiated without
    modifying client code, supporting extensible clustering and
    optimization pipelines.

    Examples
    --------
    >>> divergence = BregmanDivergenceFactory.create("squared_euclidean")
    >>> divergence
    SquaredEuclideanDivergence()

    Attributes
    ----------
    _registry : dict of str to type
        Internal mapping from divergence names to concrete
        divergence classes.
    """

    _registry: Dict[str, Type[BaseBregmanDivergence]] = {}