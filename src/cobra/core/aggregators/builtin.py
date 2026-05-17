"""
Aggregation
"""


from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import (
    AggregatorFactory,
    BaseAggregator
)

@AggregatorFactory.register("weighted_mean", "wmean")
class WeightedMeanAggregator(BaseAggregator):
    """
    Formula : y_hat = sum(y_i * w_i) / sum(w_i)
    """
    def aggregate(self, values, weights = None, **kwargs):
        vals = np.asarray(values).reshape(-1)

        if vals.size == 0:
            raise ValueError("values cannot be empty")
        
        if weights is None:
            return float(np.mean(vals))
        
        w = np.asarray(weights, dtype=float).reshape(-1)

        if w.size != vals.size:
            raise ValueError("Weights and values must have the same length.")
        
        valid = np.isfinite(w)
        vals, w = vals[valid], w[valid]

        if vals.size == 0:
            return 0.0
        
        w_sum = np.sum(w)

        if np.isclose(w_sum, 0.0):
            return float(np.mean(vals))

        return float(np.dot(vals, w) / w_sum)

    def aggregate_matrix(self, values, weights, fallback=None, **kwargs):
        V = np.asarray(values)
        W = np.asarray(weights)

        if fallback is None:
            fallback = np.mean(V)
        
        mask = np.isfinite(W)
        W = np.where(mask, W, 0.0)

        denom = np.sum(W, axis=1)
        numer = W @ V

        return np.divide(
            numer,
            denom,
            out=np.full_like(denom, fallback, dtype=float),
            where=denom != 0,
        )


@AggregatorFactory.register("weighted_vote")
class WeightedVoteAggregator(BaseAggregator):
    """
    Formula : y_hat = argmax_c sum(w_i * (y_i == c))
    """
    def aggregate(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
    ):
        vals = np.asarray(values).reshape(-1)
        if vals.size == 0:
            raise ValueError("values cannot be empty")
        
        if weights is None:
            raise ValueError("weighted_vote requires weights")
        
        w = np.asarray(weights, dtype=float).reshape(-1)

        if w.size != vals.size:
            raise ValueError("Weights and values must have the same length.")

        valid = np.isfinite(w)
        vals, w = vals[valid], w[valid]

        classes = np.unique(vals)
        scores = np.zeros(len(classes), dtype=float)

        for j, c in enumerate(classes):
            scores[j] = np.sum(w[vals == c])

        return classes[np.argmax(scores)]

    def aggregate_proba(
        self,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        **kwargs
    ):
        vals = np.asarray(values).reshape(-1)
        if vals.size == 0:
            raise ValueError("values cannot be empty")
        
        if weights is None:
            raise ValueError("weighted_vote requires weights")
        
        w = np.asarray(weights, dtype=float).reshape(-1)

        if w.size != vals.size:
            raise ValueError("Weights and values must have the same length.")

        valid = np.isfinite(w)
        vals, w = vals[valid], w[valid]

        if classes is None:
            classes = np.unique(vals)
        
        classes = np.asarray(classes)

        probs = np.zeros(len(classes), dtype=float)

        for j, c in enumerate(classes):
            probs[j] = np.sum(w[vals == c])

        total = np.sum(probs)

        if np.isclose(total, 0.0):
            return np.ones(len(classes)) / len(classes)

        return probs / total

@AggregatorFactory.register("majority_vote")
class MajorityVoteAggregator(BaseAggregator):

    def aggregate(self, values, weights=None, **kwargs):

        vals = np.asarray(values).reshape(-1)
        if vals.size == 0:
            raise ValueError("values cannot be empty")

        classes, counts = np.unique(vals, return_counts=True)

        return classes[np.argmax(counts)]