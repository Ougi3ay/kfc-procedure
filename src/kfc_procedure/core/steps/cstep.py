"""
C-step aggregation layer in the KFC pipeline.

This module implements the C-step of the KFCProcedure framework, where
divergence-specific predictions produced by the F-step are aggregated
into final outputs using a configurable combiner strategy.

Overview
--------
The C-step receives a prediction matrix:

    X ∈ R^{n_samples × n_divergences}

where each column corresponds to predictions obtained under a specific
divergence-induced model family.

The goal is to learn or apply an aggregation function:

    f: R^{n × m} → R^n

that fuses multiple divergence-aware predictions into a single final
output.

Aggregation strategies
----------------------
The following combiner families are supported:

Regression:
* mean
* weighted_mean
* stacking
* gradientcobra
* mixcobra

Classification:
* majority_vote
* stacking_classifier
* combined_classifier (COBRA-based probabilistic aggregation)

Key Idea
--------
Each divergence produces a distinct predictive view of the data.
The C-step acts as a fusion layer that reconciles these views into a
single decision or prediction.

This enables:

* ensemble learning across divergence spaces
* model selection and weighting across metrics
* meta-learning over divergence-induced predictors

Inputs
------
X : ndarray of shape (n_samples, n_divergences)
    Prediction matrix from F-step.

y : ndarray of shape (n_samples,)
    Ground truth labels (used for training combiners that require supervision).

Outputs
-------
y_pred : ndarray of shape (n_samples,)
    Final aggregated predictions.

Optional:
y_proba : ndarray of shape (n_samples, n_classes)
    Class probabilities (only for classification models supporting it).

Notes
-----
* Stateless combiners (e.g., mean, majority vote) do not require fitting.
* Learned combiners (e.g., stacking, COBRA) require training on X and y.
* The C-step is task-aware (regression vs classification).

This stage is the final fusion layer of the KFC pipeline.
"""


from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory


class CStep(BaseEstimator):
    """
    C-step: Aggregation layer for divergence-aware predictions.

    The C-step combines outputs from multiple divergence-specific models
    into a final prediction using a configurable combiner strategy.

    Parameters
    ----------
    combiner : str or BaseCombiner
        Aggregation strategy. Can be:
        - string identifier resolved via CombinerFactory
        - pre-instantiated BaseCombiner object

    combiner_params : dict, default=None
        Parameters passed to the combiner constructor.

    task : str, default="regression"
        Learning task type:
        - "regression"
        - "classification"

    random_state : int or None, default=None
        Random seed forwarded to stochastic combiners.

    Attributes
    ----------
    strategy_ : BaseCombiner
        Fitted combiner strategy instance.

    Methods
    -------
    fit(X, y)
        Fit the aggregation strategy on prediction matrix.

    predict(X)
        Return aggregated regression or class predictions.

    predict_proba(X)
        Return class probabilities (classification only).

    Notes
    -----
    The C-step operates on the output of the F-step:

        X = F_step(X_input)

    Each column of X corresponds to a divergence-specific prediction.

    The C-step performs:

        y = f(X)

    where f is a learned or rule-based aggregation function.

    Raises
    ------
    AttributeError
        If predict_proba is called on a regression task or unsupported
        combiner.
    """
    def __init__(
        self,
        combiner: Union[str, BaseCombiner],
        combiner_params: Optional[Dict] = None,
        task: str = "regression",
        random_state: Optional[int] = None,
    ):
        self.combiner = combiner
        self.combiner_params = combiner_params or {}
        self.task = task
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the combiner strategy.
        """
        self.strategy_ = self._build_combiner()
        self.strategy_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict final aggregated outputs.
        """
        check_is_fitted(self, "strategy_")
        return self.strategy_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        """
        if self.task != "classification":
            raise AttributeError("predict_proba only available for classification")

        check_is_fitted(self, "strategy_")

        if hasattr(self.strategy_, "predict_proba"):
            return self.strategy_.predict_proba(X)

        raise AttributeError(
            f"{type(self.strategy_).__name__} does not support predict_proba"
        )

    def _build_combiner(self):
        """
        Build combiner strategy from registry.
        """

        # already an instance
        if not isinstance(self.combiner, str):
            return self.combiner

        name = self.combiner.lower()

        # check existence
        if not CombinerFactory.contains(name):
            raise ValueError(
                f"'{name}' is not a valid combiner. "
                f"Available: {CombinerFactory.available()}"
            )

        # task compatibility check
        if not CombinerFactory.supports(name, self.task):
            raise ValueError(
                f"'{name}' is not valid for task='{self.task}'. "
                f"Available: {CombinerFactory.available_by_category(self.task)}"
            )
        
        params = dict(self.combiner_params)

        if "random_state" not in params:
            params["random_state"] = self.random_state

        return CombinerFactory.create(
            name,
            **params
        )
