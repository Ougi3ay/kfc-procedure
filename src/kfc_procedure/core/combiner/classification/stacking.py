"""
Stacking classifier combiner.

This module implements a stacking-based classification combiner using
a meta-classifier trained on base model predictions.
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("stacking_classifier", categories={"classification"})
class StackingClassifierCombiner(BaseCombiner):
    """
    Logistic regression stacking classifier.

    Learns a mapping from base predictions to final class labels.
    """

    def __init__(self, meta_model=None):
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(X, y)

        self._is_fitted = True
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("StackingClassifierCombiner is not fitted.")

        return self.meta_model_.predict(np.asarray(X))
