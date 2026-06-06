"""
Data splitting abstractions and factory registration infrastructure.

This module defines the common interface for dataset partitioning
strategies used throughout the COBRA framework.

Splitters are responsible for generating index partitions that divide
a dataset into training and evaluation subsets. These partitions are
subsequently used during estimator training, calibration, aggregation,
or model validation procedures.

The module also provides a dedicated factory for dynamic splitter
registration and runtime instantiation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from cobra.core.factory import BaseFactory
from cobra.core.types import SplitIndices

class BaseDataSplitter(ABC):
    """
    Abstract interface for dataset splitting strategies.

    A data splitter generates index partitions that separate a dataset
    into training and evaluation subsets. Concrete implementations may
    employ random sampling, holdout schemes, overlapping partitions,
    cross-validation protocols, temporal splits, or other strategies.

    Notes
    -----
    Splitters operate exclusively on sample indices and do not modify
    the underlying feature matrix or target vector.

    Implementations must define the :meth:`split` method and return a
    :class:`SplitIndices` object describing the resulting partition.

    Examples
    --------
    >>> splitter = RandomHoldoutSplitter(
    ...     calibration_size=0.5,
    ...     random_state=42,
    ... )
    >>> indices = splitter.split(X, y)
    >>> indices.train_idx
    >>> indices.eval_idx
    """

    @abstractmethod
    def split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
    ) -> SplitIndices:
        """
        Generate a dataset partition.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : ndarray of shape (n_samples,)
            Target values associated with the observations.

        groups : ndarray of shape (n_samples,), optional
            Group labels used by splitters that enforce
            group-aware partitioning constraints.

        Returns
        -------
        SplitIndices
            Object containing the training and evaluation indices
            produced by the splitting strategy.

        Raises
        ------
        ValueError
            If the supplied data are incompatible with the splitting
            strategy.

        Notes
        -----
        The exact partitioning behavior is implementation dependent.
        """
        raise NotImplementedError

class SplitterFactory(BaseFactory):
    """
    Factory for splitter registration and creation.

    The factory maintains a registry of available data splitting
    strategies and provides runtime instantiation based on symbolic
    names.

    Examples
    --------
    Create a registered splitter:

    >>> splitter = SplitterFactory.create(
    ...     "holdout",
    ...     calibration_size=0.3,
    ... )

    Inspect available implementations:

    >>> SplitterFactory.available()
    ['holdout', 'random_holdout', 'split_overlap']

    Notes
    -----
    The factory inherits all registration, discovery, filtering,
    and metadata capabilities from :class:`BaseFactory`.
    """
    pass
