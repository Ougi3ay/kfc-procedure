"""
C-step aggregation layer for the KFC pipeline.

The C-step aggregates the held-out prediction matrix produced by F-step
into final outputs using a configurable combiner strategy.
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.combiner.base import BaseCombiner, CombinerFactory


class CStep(BaseEstimator):
    """
    C-step: Combines divergence-level predictions into final output.

    This layer selects and applies a combiner strategy such as:
    - mean
    - weighted_mean
    - stacking
    - majority_vote
    """

    def __init__(
        self,
        combiner: Union[str, BaseCombiner],
        combiner_params: Dict = None,
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
