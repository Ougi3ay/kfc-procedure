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
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike


class BaseAggregator(ABC):
    """
    Abstract base class for COBRA aggregation strategies.

    Two execution modes are supported:

    1. scalar aggregation (mandatory)
    2. matrix aggregation (optional but recommended)

    Matrix aggregation is required for:
    - GradientCOBRA acceleration
    - batch prediction
    - CV optimization
    """

    @abstractmethod
    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        **kwargs,
    ):
        """
        Aggregate a single neighborhood.
        """
        raise NotImplementedError
    
    def aggregate_matrix(
        self,
        values: ArrayLike,
        weights: ArrayLike,
        fallback: float | ArrayLike = 0.0,
        **kwargs,
    ) -> np.ndarray:

        V = np.asarray(values)
        W = np.asarray(weights)

        # ensure 2D safety
        if V.ndim == 1:
            V = np.tile(V, (W.shape[0], 1))

        mask = np.isfinite(W)
        W = np.where(mask, W, 0.0)

        denom = np.sum(W, axis=1)
        numer = W @ V

        # fallback logic
        if np.isscalar(fallback):
            fallback_vec = np.full(numer.shape, fallback, dtype=float)
        else:
            fallback_vec = np.asarray(fallback)

        out = np.divide(
            numer,
            denom[:, None] if numer.ndim == 2 else denom,
            out=fallback_vec,
            where=denom[:, None] != 0 if numer.ndim == 2 else denom != 0,
        )

        return out

    def aggregate_proba(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        **kwargs,
    ):
        raise NotImplementedError

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