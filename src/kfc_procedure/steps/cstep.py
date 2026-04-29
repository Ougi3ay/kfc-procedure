"""
CStep:
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.aggregations.base import AggregationClassifierFactory, AggregationRegressorFactory, BaseAggregation


class BaseCStep(ABC, BaseEstimator):

    strategy_: BaseAggregation

    @abstractmethod
    def fit(self, predictions: np.ndarray, y: np.ndarray, *args, **kwargs) -> "BaseCStep":
        """Fit the aggregation strategy on the held-out prediction matrix."""
        ...
    
    @abstractmethod
    def predict(self, predictions: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Aggregate the prediction matrix into a single output vector."""
        ... 

class CStep(BaseCStep):
    """
    A concrete implementation of the CStep.
    Parameters
    ----------
    config : dict
        Must include 'name' key specifying the strategy to use, 
        and optionally a 'params' dict for strategy hyperparameters.
    """
    def __init__(
        self,
        config: Dict,
    ):
        self.config = config

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "CStep":
        name = self.config.get("name")
        params = self.config.get("params", {})
        if name in AggregationRegressorFactory.available():
            self.strategy_ = AggregationRegressorFactory.create(name, **params)
        elif name in AggregationClassifierFactory.available():
            self.strategy_ = AggregationClassifierFactory.create(name, **params)
        else:
            raise ValueError(
                f"Unknown aggregation strategy: {name!r}. "
                f"Available regression: {AggregationRegressorFactory.available()} | "
                f"classification: {AggregationClassifierFactory.available()}"
            )
        self.strategy_.fit(predictions, y)
        return self
    
    def predict(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        check_is_fitted(self, "strategy_")
        return self.strategy_.predict(predictions, **kwargs)
    
    def predict_proba(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        check_is_fitted(self, "strategy_")
        return self.strategy_.predict_proba(predictions, **kwargs)
    