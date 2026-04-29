"""
kfc_procedure.aggregations
--------------------------
C-step aggregation strategies.

Regression  (task="regression")
    "mean"          –  row-wise mean
    "weighted_mean" –  OLS-learned weighted average
    "stacking"      –  meta-regressor

Classification  (task="classification")
    "majority_vote" –  row-wise majority vote
    "stacking"      –  meta-classifier
"""
from kfc_procedure.core.aggregations.base import BaseAggregation, BaseAggregationRegressor, BaseAggregationClassifier
from kfc_procedure.core.aggregations.regression import (
    MeanAggregation,
    StackingAggregation,
    WeightedMeanAggregation,
)
from kfc_procedure.core.aggregations.classification import (
    MajorityVoteAggregation,
    StackingClassifierAggregation,
)

__all__ = [
    "BaseAggregation",
    "BaseAggregationRegressor",
    "BaseAggregationClassifier",
    "MeanAggregation",
    "WeightedMeanAggregation",
    "StackingAggregation",
    "MajorityVoteAggregation",
    "StackingClassifierAggregation",
]
