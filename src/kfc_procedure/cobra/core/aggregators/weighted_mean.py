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
    Weighted mean aggregation (regression).

    Formula:
        y = (Σ w_i x_i) / Σ w_i
    """

    def aggregate(self, values, weights=None, fallback=None, **kwargs):
        V = np.asarray(values, dtype=float).reshape(-1)

        if V.size == 0:
            raise ValueError("values cannot be empty")

        if fallback is None:
            fallback = float(np.mean(V))

        if weights is None:
            return float(np.mean(V))

        W = np.asarray(weights, dtype=float).reshape(-1)
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        denom = np.sum(W)

        if np.isclose(denom, 0.0):
            return fallback

        return float(np.dot(W, V) / denom)

    def aggregate_proba(self, values, weights=None, classes=None, **kwargs):
        """
        Optional: only meaningful if values already represent probabilities.
        """
        V = np.asarray(values, dtype=float)

        if weights is None:
            return np.mean(V, axis=0)

        W = np.asarray(weights, dtype=float)
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        W = W / (np.sum(W) + 1e-12)

        return np.sum(W[:, None] * V, axis=0)
