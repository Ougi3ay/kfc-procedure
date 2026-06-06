"""
Majority vote combiner (classification).

This module implements hard voting across base classifier predictions.
The final label is selected as the most frequent prediction per sample.
"""

import numpy as np
from collections import Counter

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("majority_vote", categories={"classification"})
class MajorityVoteCombiner(BaseCombiner):
    """
    Hard voting ensemble combiner.

    Each sample is assigned the most frequent label among base models.
    """

    def fit(self, X: np.ndarray, y=None):
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        n_samples = X.shape[0]
        outputs = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
            row = X[i]
            outputs[i] = Counter(row).most_common(1)[0][0]

        return outputs