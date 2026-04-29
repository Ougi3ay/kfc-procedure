"""
kfc_procedure.aggregations.classification
------------------------------------------
Classification aggregation strategies.

    "majority_vote" –  row-wise mode across all local classifier predictions
    "stacking"      –  meta-classifier trained on the prediction matrix
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from cobra.combine_classifier import CombineClassifier
from kfc_procedure.core.aggregations.base import AggregationClassifierFactory, BaseAggregationClassifier

@AggregationClassifierFactory.register("majority_vote")
class MajorityVoteAggregation(BaseAggregationClassifier):
    """Row-wise majority vote. Stateless — fit only records seen classes."""
    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "MajorityVoteAggregation":
        self.classes_ = np.unique(y)
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return stats.mode(predictions.astype(int), axis=1, keepdims=False).mode

@AggregationClassifierFactory.register("stacking")
class StackingClassifierAggregation(BaseAggregationClassifier):
    """
    Meta-classifier stacking.

    Parameters
    ----------
    meta_estimator : sklearn classifier | None
        Defaults to LogisticRegression when None.
    """

    def __init__(self, meta_estimator=None) -> None:
        self.meta_estimator = meta_estimator

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "StackingClassifierAggregation":
        if self.meta_estimator is not None:
            self.meta_ = self.meta_estimator
        else:
            from sklearn.linear_model import LogisticRegression
            self.meta_ = LogisticRegression(max_iter=1000)
        self.meta_.fit(predictions, y)
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return self.meta_.predict(predictions)

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Return class probabilities. Requires meta_estimator to support it."""
        if not hasattr(self.meta_, "predict_proba"):
            raise AttributeError(
                f"{type(self.meta_).__name__} does not implement predict_proba."
            )
        return self.meta_.predict_proba(predictions)

@AggregationClassifierFactory.register("combine_classifier", "cc")
class CombineClassifierAggregation(BaseAggregationClassifier):
    """
    GradientCOBRA for classification.

    Parameters
    ----------
    kwargs : passed to GradientCOBRA constructor.  See its docstring for details.
    """

    def __init__(self, **kwargs) -> None:
        self.gradientcobra_ = CombineClassifier(**kwargs)

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> "CombineClassifierAggregation":
        self.gradientcobra_.fit(predictions, y, as_predictions=True)
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return self.gradientcobra_.predict(predictions)
