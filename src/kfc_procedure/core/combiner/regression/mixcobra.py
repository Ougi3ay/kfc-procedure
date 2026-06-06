"""
MixCOBRA combiner.

This module wraps MixCOBRARegressor into a sklearn-style combiner.
"""

import numpy as np
from cobra.mixcobra import MixCOBRARegressor

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("mixcobra", categories={"regression"})
class MixCOBRACombiner(BaseCombiner):
    """
    MixCOBRA-based regression combiner.
    """

    def __init__(self, **cobra_params):
        self.cobra = MixCOBRARegressor(**cobra_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.cobra.fit(X, y, as_predictions=True)
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        return self.cobra.predict(X)
