"""
Aggregation package for COBRA framework.

This package provides the final aggregation stage of the COBRA pipeline.
"""

from .base import BaseAggregator, AggregatorFactory
from .weighted_mean import WeightedMeanAggregator
from .weighted_vote import WeightedVoteAggregator

__all__ = [
    "BaseAggregator",
    "AggregatorFactory",
    "WeightedMeanAggregator",
    "WeightedVoteAggregator"
]