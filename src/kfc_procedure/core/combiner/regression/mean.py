"""
Mean combiner (regression).

This module implements a simple row-wise arithmetic mean combiner
for ensemble regression. It aggregates base model predictions by
computing the unweighted average across models.

Mathematically:

    f(x_1, ..., x_K) = (1/K) Σ x_k

This is a stateless combiner and does not require training.
"""

import numpy as np
from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("mean", categories={"regression"})
class MeanCombiner(BaseCombiner):
    """
    Row-wise mean combiner for regression ensembles.

    This combiner computes the arithmetic mean of base model predictions
    for each sample.

    No training is required.

    Methods
    -------
    fit(X, y=None)
        Stateless training step (returns self).
    combine(X)
        Returns mean prediction across models.
    """

    def fit(self, X: np.ndarray, y=None):
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        return np.mean(X, axis=1)
