"""
kfc_procedure.aggregations.regression
---------------------------------------
Regression aggregation strategies.

    "mean"          –  row-wise mean (stateless)
    "weighted_mean" –  OLS-learned weighted average
    "stacking"      –  meta-regressor trained on the prediction matrix
"""
from __future__ import annotations

import numpy as np

from kfc_procedure.core.aggregations.base import AggregationRegressorFactory, BaseAggregationRegressor
from cobra.gradientcobra import GradientCOBRA

@AggregationRegressorFactory.register("mean")
class MeanAggregation(BaseAggregationRegressor):
    """Row-wise mean of all local predictions. Stateless — fit is a no-op."""
    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "MeanAggregation":
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return predictions.mean(axis=1)


@AggregationRegressorFactory.register("weighted_mean")
class WeightedMeanAggregation(BaseAggregationRegressor):
    """OLS-learned weighted average: solves predictions @ w ≈ y."""

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "WeightedMeanAggregation":
        self.weights_, *_ = np.linalg.lstsq(predictions, y, rcond=None)
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return predictions @ self.weights_


@AggregationRegressorFactory.register("stacking")
class StackingAggregation(BaseAggregationRegressor):
    """
    Meta-regressor stacking.

    Parameters
    ----------
    meta_estimator : sklearn regressor | None
        Defaults to LinearRegression when None.
    """

    def __init__(self, meta_estimator=None) -> None:
        self.meta_estimator = meta_estimator

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "StackingAggregation":
        if self.meta_estimator is not None:
            self.meta_ = self.meta_estimator
        else:
            from sklearn.linear_model import LinearRegression
            self.meta_ = LinearRegression()
        self.meta_.fit(predictions, y)
        return self

    def predict(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        return self.meta_.predict(predictions, **kwargs)

@AggregationRegressorFactory.register("gradientcobra")
class GradientCobraAggregation(BaseAggregationRegressor):
    """
    GradientCobra
    """

    def __init__(self, **kwargs) -> None:
        self.gradientcobra_ = GradientCOBRA(**kwargs)
    
    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "GradientCobraAggregation":
        self.gradientcobra_.fit(predictions, y, as_predictions=True)
        return self
    
    def predict(self, predictions: np.ndarray, bandwidth: float) -> np.ndarray:
        return self.gradientcobra_.predict(predictions, bandwidth=bandwidth)
    
