"""
Local models (Lm)
"""

from __future__ import annotations

from abc import ABC
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from kfc_procedure.core.factory import BaseFactory


class BaseLocalModel(ABC, BaseEstimator):
    """
    Base class for local models (F-step).
    """
    ...

class BaseLocalModelRegressor(BaseLocalModel, RegressorMixin):
    """
    Base class for regression local models.
    """
    ...

class BaseLocalModelClassifier(BaseLocalModel, ClassifierMixin):
    """
    Base class for classification local models.
    """
    ...

class LocalModelRegressorFactory(BaseFactory):
    """
    Factory for regression local models.
    """
    _registry: dict[str, type[BaseLocalModelRegressor]] = {}

class LocalModelClassifierFactory(BaseFactory):
    """
    Factory for classification local models.
    """
    _registry: dict[str, type[BaseLocalModelClassifier]] = {}
