"""
Aggregation module for final prediction consensus.

This module defines the aggregation stage of the COBRA pipeline, where
neighbor target values (and optional weights) are converted into a single
final prediction.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output


Purpose
-------
After kernel evaluation identifies relevant neighbors and computes
their associated weights, the aggregation stage combines those neighbor
target values into one scalar prediction.

This is the final consensus step before producing model output.

Typical aggregation strategies include:

- mean aggregation
- weighted mean aggregation
- median aggregation
- voting-based aggregation
- robust aggregation rules

By separating aggregation logic into dedicated strategies, the framework
becomes:

- modular
- easily extensible
- compatible with optimization workflows
- simple to test and benchmark

Examples
--------
>>> @AggregatorFactory.register("mean")
... class MeanAggregator(BaseAggregator):
...     def aggregate(self, values, weights=None):
...         return float(np.mean(values))

>>> aggregator = AggregatorFactory.create("mean")
>>> pred = aggregator.aggregate([1.2, 1.5, 1.8])
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseAggregator(ABC):
    """
    Abstract base class for aggregation strategies.

    Aggregators define how multiple neighbor target values are combined
    into a single scalar prediction.

    This is the final prediction step of the COBRA architecture.

    Parameters
    ----------
    values : ArrayLike
        Neighbor target values selected by the kernel stage.

    weights : ArrayLike or None, default=None
        Optional weights associated with each value.

    Notes
    -----
    Subclasses must implement the ``aggregate()`` method.

    Some strategies may ignore weights (e.g., median),
    while others depend heavily on them (e.g., weighted mean).

    Examples
    --------
    >>> class MeanAggregator(BaseAggregator):
    ...     def aggregate(self, values, weights=None):
    ...         return float(np.mean(values))
    """

    @abstractmethod
    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        **kwargs,
    ):
        """
        Aggregate values into a single scalar prediction.

        Parameters
        ----------
        values : ArrayLike
            Target values from selected neighbors.

        weights : ArrayLike or None, default=None
            Optional weights for weighted aggregation.

        Returns
        -------
        Any : 
            Final aggregated prediction.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> aggregator.aggregate([1.0, 2.0, 3.0])
        2.0
        """
        raise NotImplementedError


def _as_1d(values: ArrayLike) -> np.ndarray:
    """
    Convert input values into a validated 1D float array.

    This helper ensures stable aggregation by:

    - converting values to NumPy arrays
    - forcing float dtype
    - flattening into 1D shape
    - validating non-empty input

    Parameters
    ----------
    values : ArrayLike
        Input values to normalize.

    Returns
    -------
    np.ndarray
        Flattened 1D float array.

    Raises
    ------
    ValueError
        If the input is empty.

    Examples
    --------
    >>> _as_1d([[1, 2], [3, 4]])
    array([1., 2., 3., 4.])
    """
    arr = np.asarray(values, dtype=float).reshape(-1)

    if arr.size == 0:
        raise ValueError(
            "Cannot aggregate an empty set of values."
        )

    return arr


class AggregatorFactory(BaseFactory):
    """
    Factory for ``BaseAggregator`` implementations.

    This registry-based factory enables dynamic creation of aggregation
    strategies using string identifiers.

    It is especially useful for:

    - YAML-based model configuration
    - experiment pipelines
    - hyperparameter search systems
    - benchmarking multiple aggregation rules

    Examples
    --------
    >>> aggregator = AggregatorFactory.create("mean")

    >>> prediction = aggregator.aggregate(
    ...     values=[1.2, 1.4, 1.8]
    ... )
    """