"""
GradientCOBRA combiner.

This module wraps GradientCOBRA into a sklearn-style combiner.
"""

import numpy as np
from kfc_procedure.cobra.gradientcobra import GradientCOBRA

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("gradientcobra", categories={"regression"})
class GradientCOBRACombiner(BaseCombiner):
    """
    GradientCOBRA-based regression combiner.
    """

    def __init__(self, **cobra_params):
        self.cobra = GradientCOBRA(**cobra_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y, as_predictions=True)
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)
