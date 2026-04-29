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


@AggregatorFactory.register("majority_vote", "vote")
class MajorityVoteAggregator(BaseAggregator):
    """
    Majority vote aggregator.

    This aggregator selects the most frequent class label among
    candidate values.

    Mathematical form
    -----------------
    prediction = argmax(count(label))

    Commonly used for classification tasks.

    Notes
    -----
    This implementation ignores weights.

    In case of ties, NumPy's ``argmax()`` returns the first occurrence.

    Examples
    --------
    >>> aggregator = MajorityVoteAggregator()

    >>> aggregator.aggregate([1, 1, 2, 1, 3])
    1.0
    """

    def __init__(self, classes=None):
        self.classes_ = np.asarray(classes) if classes is not None else None

    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        return_proba: bool = False,
    ) -> float | dict:
        """
        Compute majority vote.

        Parameters
        ----------
        values : ArrayLike
            Candidate class labels.

        weights : ArrayLike or None, default=None
            Ignored in this implementation.

        return_proba : bool, default=False
            If True, return probability distribution instead of single label.

        Returns
        -------
        float | dict
            Most frequent class label or probability distribution.

        Raises
        ------
        ValueError
            If values are empty.
        """
        vals = np.asarray(values).reshape(-1)

        if vals.size == 0:
            raise ValueError(
                "Cannot aggregate an empty set of values."
            )

        uniq, counts = np.unique(vals, return_counts=True)
        
        if return_proba:
            if self.classes_ is None:
                raise ValueError(
                    "classes_ must be defined for predict_proba()."
                )

            proba = np.zeros(len(self.classes_))
            total = np.sum(counts)

            count_map = dict(zip(uniq, counts))

            for i, cls in enumerate(self.classes_):
                proba[i] = count_map.get(cls, 0) / total

            return proba

        return float(uniq[np.argmax(counts)])
