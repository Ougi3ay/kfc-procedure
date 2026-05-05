"""
C-step aggregation layer for the KFC pipeline.

The C-step aggregates the held-out prediction matrix produced by F-step
into final outputs. It supports both regression and classification
aggregation strategies and can be configured by name through a dictionary.
"""

from __future__ import annotations
from typing import Dict, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.aggregations.base import (
    AggregationClassifierFactory,
    AggregationRegressorFactory,
    BaseAggregationClassifier,
    BaseAggregationRegressor
)


class CStep(BaseEstimator):
    """Aggregation step.

    The C-step combines divergence-specific predictions into a final ensemble
    prediction.

    Parameters
    ----------
    aggregation : str or BaseAggregationRegressor or BaseAggregationClassifier
        Aggregation strategy identifier or instance.
    aggregation_param : dict
        Parameters passed to aggregation strategy builder.
    task : str
        Either ``"regression"`` or ``"classification"``.
    """

    def __init__(
        self,
        aggregation: Union[str, BaseAggregationRegressor, BaseAggregationClassifier],
        aggregation_param: Dict,
        task: str,
    ):
        self.aggregation = aggregation
        self.aggregation_param = aggregation_param
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the aggregation strategy.

        Parameters
        ----------
        X : ndarray
            Feature matrix of divergence-level predictions.
        y : ndarray
            Target values.

        Returns
        -------
        self : CStep
            The fitted C-step instance.
        """
        self.strategy_ = self._build_aggregation()
        self.strategy_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict final outputs using the aggregation strategy.

        Parameters
        ----------
        X : ndarray
            Feature matrix of divergence-level predictions.

        Returns
        -------
        ndarray
            Aggregated predictions.
        """
        check_is_fitted(self, "strategy_")
        return self.strategy_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the aggregation strategy.

        Parameters
        ----------
        X : ndarray
            Feature matrix of divergence-level predictions.

        Returns
        -------
        ndarray
            Aggregated class probabilities.

        Raises
        ------
        AttributeError
            If the task is not classification or the aggregation strategy does
            not support prediction probabilities.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba only available for classification")

        if hasattr(self.strategy_, "predict_proba"):
            return self.strategy_.predict_proba(X)

        raise AttributeError("Aggregation does not support predict_proba")

    def _build_aggregation(self):
        """
        Build aggregation strategy based on task type.

        Returns
        -------
        BaseEstimator
            Aggregation strategy instance.

        Raises
        ------
        ValueError
            If aggregation name is invalid for the selected task.
        """

        if not isinstance(self.aggregation, str):
            return self.aggregation

        name = self.aggregation

        if self.task == "regression":
            if name in AggregationRegressorFactory.available():
                return AggregationRegressorFactory.create(
                    name, **self.aggregation_param
                )

            raise ValueError(
                f"'{name}' is not a valid REGRESSION aggregation. "
                f"Available: {AggregationRegressorFactory.available()}"
            )

        if self.task == "classification":
            if name in AggregationClassifierFactory.available():
                return AggregationClassifierFactory.create(
                    name, **self.aggregation_param
                )

            raise ValueError(
                f"'{name}' is not a valid CLASSIFICATION aggregation. "
                f"Available: {AggregationClassifierFactory.available()}"
            )

        raise ValueError(f"Unknown task '{self.task}'")
