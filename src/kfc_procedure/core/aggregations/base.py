"""
Base aggregation class
"""

from __future__ import annotations
from abc import ABC
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from kfc_procedure.core.factory import BaseFactory



class BaseAggregation(ABC, BaseEstimator):
    """
    Base class for all aggregation operations
    """
    ...

class BaseAggregationRegressor(BaseAggregation, RegressorMixin):
    """
    Base class for regression aggregation strategies
    """

class BaseAggregationClassifier(BaseAggregation, ClassifierMixin):
    """
    Base class for classification aggregation strategies
    """
    ...

class AggregationRegressorFactory(BaseFactory):
    """
    Factory for regression aggregation strategies
    """
    _registry = {}

class AggregationClassifierFactory(BaseFactory):
    """
    Factory for classification aggregation strategies
    """
    _registry = {}