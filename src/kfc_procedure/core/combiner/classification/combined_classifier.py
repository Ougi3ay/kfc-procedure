"""
COBRA combined classifier wrapper.

This module integrates COBRA CombinedClassifier into the combiner API.
"""

import numpy as np
from cobra.combined_classifier import CombinedClassifier

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("combined_classifier", categories={"classification"})
class CobraClassifierCombiner(BaseCombiner):
    """
    COBRA-based classifier combiner.

    Supports probability prediction via COBRA aggregation.
    """

    def __init__(self, **cobra_params):
        self.cobra = CombinedClassifier(**cobra_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y, as_predictions=True)
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict_proba(X)