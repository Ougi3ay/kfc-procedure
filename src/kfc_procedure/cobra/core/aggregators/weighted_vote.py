"""
Weighted Vote Aggregator (COBRA)

Classification aggregation using weighted voting.

Mathematical form:
------------------
Single:
    y = argmax_c Σ w_i * 1(v_i == c)

Batch:
    score(q, c) = Σ_i W[q,i] * 1(v_i == c)

Returns:
    argmax_c score(q, c)
"""

from __future__ import annotations

import numpy as np
from .base import BaseAggregator, AggregatorFactory


@AggregatorFactory.register("weighted_vote", "wv")
class WeightedVoteAggregator(BaseAggregator):
    """
    Fully vectorized weighted majority vote.

    Uses one-hot encoding for class aggregation.
    """

    # ======================================================
    # SINGLE QUERY
    # ======================================================
    def aggregate(self, values, weights=None, **kwargs):
        V = np.asarray(values).reshape(-1)

        if V.size == 0:
            raise ValueError("values cannot be empty")

        # fallback: majority vote
        if weights is None:
            classes, counts = np.unique(V, return_counts=True)
            return classes[np.argmax(counts)]

        W = np.asarray(weights, dtype=float).reshape(-1)

        if W.size != V.size:
            raise ValueError("weights and values must match length")

        mask = np.isfinite(W)
        V, W = V[mask], W[mask]

        classes = np.unique(V)
        one_hot = (V[:, None] == classes[None, :]).astype(float)

        scores = W @ one_hot  # (C,)

        return classes[np.argmax(scores)]

    # ======================================================
    # BATCH VERSION (FULLY VECTORIZED)
    # ======================================================
    def aggregate_matrix(self, values, weights, **kwargs):
        """
        Batch weighted voting.

        Parameters
        ----------
        values : np.ndarray
            Shape (n_models,)

        weights : np.ndarray
            Shape (n_queries, n_models)

        Returns
        -------
        np.ndarray
            Shape (n_queries,)
        """
        V = np.asarray(values).reshape(-1)
        W = np.asarray(weights, dtype=float)

        if W.ndim != 2:
            raise ValueError(f"weights must be 2D, got {W.shape}")

        classes = np.unique(V)

        # (M, C)
        one_hot = (V[:, None] == classes[None, :]).astype(float)

        # (Q, M) @ (M, C) -> (Q, C)
        scores = W @ one_hot

        return classes[np.argmax(scores, axis=1)]

    # ======================================================
    # PROBABILITY OUTPUT
    # ======================================================
    def aggregate_proba(self, values, weights=None, classes=None, **kwargs):
        V = np.asarray(values).reshape(-1)

        if classes is None:
            classes = np.unique(V)

        classes = np.asarray(classes)

        one_hot = (V[:, None] == classes[None, :]).astype(float)
        probs = one_hot.mean(axis=0)

        return probs / (np.sum(probs) + 1e-12)

    def aggregate_proba_batch(self, values, weights, classes=None, **kwargs):
        V = np.asarray(values).reshape(-1)
        W = np.asarray(weights, dtype=float)

        if classes is None:
            classes = np.unique(V)

        classes = np.asarray(classes)

        one_hot = (V[:, None] == classes[None, :]).astype(float)

        scores = W @ one_hot
        return scores / (np.sum(scores, axis=1, keepdims=True) + 1e-12)
