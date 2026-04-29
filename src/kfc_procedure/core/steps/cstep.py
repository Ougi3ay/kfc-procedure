"""
C-step aggregation layer for the KFC pipeline.

The C-step aggregates the held-out prediction matrix produced by F-step
into final outputs. It supports both regression and classification
aggregation strategies and can be configured by name through a dictionary.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.aggregations.base import AggregationClassifierFactory, AggregationRegressorFactory, BaseAggregation


class BaseCStep(ABC, BaseEstimator):
    """
    Abstract aggregation stage interface for the C-step.
    """

    strategy_: BaseAggregation

    @abstractmethod
    def fit(self, predictions: np.ndarray, y: np.ndarray, *args, **kwargs) -> "BaseCStep":
        """
        Fit the aggregation strategy on the held-out prediction matrix.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction matrix from the F-step.

        y : np.ndarray
            Target values or labels to fit the aggregator.

        Returns
        -------
        BaseCStep
            Fitted aggregation stage.
        """
        ...
    
    @abstractmethod
    def predict(self, predictions: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Aggregate the prediction matrix into a single output vector.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction matrix from the F-step.

        Returns
        -------
        np.ndarray
            Final predictions.
        """
        ... 

class CStep(BaseCStep):
    """
    Configuration wrapper for a concrete aggregation strategy.

    Parameters
    ----------
    config : dict
        Dictionary containing the aggregator name and optional parameters.

        Expected keys:

        - name : str
            Aggregator alias.
        - params : dict, optional
            Hyperparameters passed to the aggregation implementation.
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
        """
        Predict class probabilities using the fitted aggregation strategy.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction matrix from the F-step.

        Returns
        -------
        np.ndarray
            Class probability estimates.
        """
        check_is_fitted(self, "strategy_")
        return self.strategy_.predict_proba(predictions, **kwargs)
    