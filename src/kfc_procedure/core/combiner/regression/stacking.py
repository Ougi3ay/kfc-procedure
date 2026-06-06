"""
Stacking combiner for regression.

This module implements a stacking ensemble combiner that learns a
meta-regressor over base model predictions.

Model:

    f(X) -> y
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("stacking_regressor", categories={"regression"})
class StackingRegressorCombiner(BaseCombiner):
    """
    Stacking combiner using a regression meta-model.

    The meta-model learns to map base predictions to target values.

    Parameters
    ----------
    meta_model : estimator, default=LinearRegression()
        Regression model used as meta-learner.
    """

    def __init__(self, meta_model=None):
        self.meta_model = meta_model or LinearRegression()
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
            raise RuntimeError("StackingRegressorCombiner is not fitted.")

        return self.meta_model_.predict(np.asarray(X))
