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
from kfc_procedure.core.combiner.base import BaseCombiner
from kfc_procedure.core.combiner.builtin import (
    MeanCombiner,
    WeightedMeanCombiner,
    StackingCombiner,
    StackingClassifierCombiner,
    MajorityVoteCombiner
)

__all__ = [
    "BaseCombiner",
    "MeanCombiner",
    "WeightedMeanCombiner",
    "StackingCombiner",
    "StackingClassifierCombiner",
    "MajorityVoteCombiner"
]
