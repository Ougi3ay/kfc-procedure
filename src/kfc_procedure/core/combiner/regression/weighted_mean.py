"""
Weighted mean combiner (regression).

Learns optimal linear weights using Ordinary Least Squares (OLS).

Model:

    y ≈ Xw

where:
    X ∈ R^{n × K} is the prediction matrix
    w ∈ R^K are learned weights
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner import CombinerFactory


@CombinerFactory.register("weighted_mean", categories={"regression"})
class WeightedMeanCombiner(BaseCombiner):
    """
    Linear regression-based weighted combiner.

    Learns optimal weights for base model predictions.

    Parameters
    ----------
    fit_intercept : bool, default=False
        Whether to fit intercept in linear model.
    """

    def __init__(self, fit_intercept: bool = False):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.shape}")

        self.model.fit(X, y)
        return self

    def combine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.model.predict(X)
