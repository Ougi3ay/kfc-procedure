"""
Built-in aggregation strategies for COBRA-style consensus.

This module provides concrete implementations of ``BaseAggregator`` used
in the final aggregation stage of the COBRA pipeline.

Implemented aggregators
-----------------------

1. SimpleMeanAggregator

    Computes the arithmetic mean of neighbor target values.

    Logic:
        prediction = mean(values)

2. WeightedMeanAggregator

    Computes a weighted mean using kernel-generated weights.

    Logic:
        prediction = sum(values * weights) / sum(weights)

    If weights are missing or invalid, it safely falls back to
    simple mean aggregation.

3. MajorityVoteAggregator

    Computes the most frequent class label among candidate values.

    Logic:
        prediction = most frequent label

    Commonly used for classification settings.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

These implementations are automatically registered using
``AggregatorFactory.register()`` and can be instantiated dynamically.

Examples
--------
>>> aggregator = AggregatorFactory.create("mean")
>>> pred = aggregator.aggregate([1.2, 1.5, 1.8])

>>> aggregator = AggregatorFactory.create("weighted_mean")
>>> pred = aggregator.aggregate(
...     values=[1.2, 1.5, 1.8],
...     weights=[0.2, 0.5, 0.3]
... )

>>> aggregator = AggregatorFactory.create("majority_vote")
>>> pred = aggregator.aggregate([1, 1, 2, 1, 3])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import (
    AggregatorFactory,
    BaseAggregator,
    _as_1d,
)


@AggregatorFactory.register("mean", "simple_mean")
class SimpleMeanAggregator(BaseAggregator):
    """
    Arithmetic mean aggregator.

    This aggregator computes the simple arithmetic mean of all
    candidate neighbor values.

    Mathematical form
    -----------------
    prediction = mean(values)

    Notes
    -----
    This method ignores weights even if they are provided.

    Commonly used in regression settings where all neighbors are treated
    equally.

    Examples
    --------
    >>> aggregator = SimpleMeanAggregator()
    >>> aggregator.aggregate([1.0, 2.0, 3.0])
    2.0
    """

    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> float:
        """
        Compute arithmetic mean.

        Parameters
        ----------
        values : ArrayLike
            Candidate target values.

        weights : ArrayLike or None, default=None
            Ignored in this implementation.

        Returns
        -------
        float
            Mean of input values.
        """
        vals = _as_1d(values)
        return float(np.mean(vals))


@AggregatorFactory.register("weighted_mean", "wmean")
class WeightedMeanAggregator(BaseAggregator):
    """
    Weighted mean aggregator.

    This aggregator computes a weighted average using kernel-generated
    weights.

    Mathematical form
    -----------------
    prediction = sum(values × weights) / sum(weights)

    If weights are missing or their total weight is zero,
    the method safely falls back to simple mean aggregation.

    Notes
    -----
    This is one of the most common aggregation strategies in
    kernel-based COBRA methods.

    Examples
    --------
    >>> aggregator = WeightedMeanAggregator()

    >>> aggregator.aggregate(
    ...     values=[1.0, 2.0, 3.0],
    ...     weights=[0.2, 0.5, 0.3]
    ... )
    """

    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> float:
        """
        Compute weighted mean with safe fallback.

        Parameters
        ----------
        values : ArrayLike
            Candidate target values.

        weights : ArrayLike or None, default=None
            Optional weights for weighted aggregation.

        Returns
        -------
        float
            Weighted mean prediction.

        Raises
        ------
        ValueError
            If weights and values have different lengths.
        """
        vals = _as_1d(values)

        if weights is None:
            return float(np.mean(vals))

        w = np.asarray(weights, dtype=float).reshape(-1)

        if w.size != vals.size:
            raise ValueError(
                "weights and values must have the same length."
            )

        w_sum = np.sum(w)

        if np.isclose(w_sum, 0.0):
            return float(np.mean(vals))

        return float(np.sum(vals * w) / w_sum)


@AggregatorFactory.register("weighted_vote")
class WeightedVoteAggregator(BaseAggregator):
    """
    Formula : y_hat = argmax_c * sum_i w_i * 1(y_i == c) 
    """
    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
    ):
        values = np.asarray(values)
        weights = np.asarray(weights)

        classes = np.unique(values)
        scores = np.zeros(len(classes), dtype=float)

        for j, c in enumerate(classes):
            scores[j] = np.sum(weights[values == c])

        return classes[np.argmax(scores)]

    def aggregate_proba(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        **kwargs
    ):
        values = np.asarray(values)
        weights = np.asarray(weights)

        if classes is None:
            classes = np.unique(values)
        else:
            classes = np.asarray(classes)

        proba = np.zeros(len(classes), dtype=float)

        for j, c in enumerate(classes):
            proba[j] = np.sum(weights[values == c])

        proba_sum = np.sum(proba)

        if proba_sum <= 0:
            return np.ones(len(classes)) / len(classes)

        return proba / proba_sum

@AggregatorFactory.register("majority_vote")
class MajorityVoteAggregator(BaseAggregator):

    def aggregate(self, values, weights=None, **kwargs):

        values = np.asarray(values)

        classes, counts = np.unique(values, return_counts=True)

        return classes[np.argmax(counts)]