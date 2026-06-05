"""
Weighted Mean Aggregator (COBRA)

Regression aggregation using normalized weights.

Formula:
    y = (W @ V) / sum(W)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import AggregatorFactory, BaseAggregator

@AggregatorFactory.register("weighted_mean", "wm")
class WeightedMeanAggregator(BaseAggregator):
    """
    Weighted mean aggregation for regression tasks.

    This is the standard COBRA regression aggregator using
    kernel-derived weights.
    """

    def aggregate(self, values, weights=None, **kwargs):
        V = np.asarray(values, dtype=float).reshape(-1)

        if V.size == 0:
            raise ValueError("values cannot be empty")

        if weights is None:
            return float(np.mean(V))

        W = np.asarray(weights, dtype=float).reshape(-1)

        W = np.where(np.isfinite(W), W, 0.0)

        denom = np.sum(W)

        if np.isclose(denom, 0.0):
            return float(np.mean(V))

        return float(np.dot(W, V) / denom)

    def aggregate_batch(self, values, weights, fallback=None, **kwargs):
        V = np.asarray(values, dtype=float)
        W = np.asarray(weights, dtype=float)

        if fallback is None:
            fallback = float(np.mean(V))

        W = np.where(np.isfinite(W), W, 0.0)

        denom = np.sum(W, axis=1)
        numer = W @ V

        if V.ndim == 1:
            return np.divide(
                numer,
                denom,
                out=np.full_like(denom, fallback, dtype=float),
                where=denom != 0,
            )

        return np.divide(
            numer,
            denom[:, None],
            out=np.full_like(numer, fallback, dtype=float),
            where=denom[:, None] != 0,
        )

    def aggregate_proba(self, values, weights=None, classes=None, **kwargs):
        V = np.asarray(values, dtype=float)

        if weights is None:
            return np.mean(V, axis=0)

        W = np.asarray(weights, dtype=float)
        W = np.where(np.isfinite(W), W, 0.0)

        W = W / (np.sum(W) + 1e-12)

        return np.sum(W[:, None] * V, axis=0)

    def aggregate_proba_batch(self, values, weights, classes=None, **kwargs):
        """
        Correct vectorized probability aggregation.

        values : (n_models, n_classes)
        weights: (n_queries, n_models)

        returns : (n_queries, n_classes)
        """
        V = np.asarray(values, dtype=float)
        W = np.asarray(weights, dtype=float)

        W = np.where(np.isfinite(W), W, 0.0)

        denom = np.sum(W, axis=1, keepdims=True) + 1e-12
        W = W / denom

        return W @ V
