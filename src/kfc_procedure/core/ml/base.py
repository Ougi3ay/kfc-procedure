"""
Local model interface and factory definitions for F-step.

The local model layer provides base estimator classes and factories that are
resolved by name when configuring the F-step.
"""

from __future__ import annotations

from abc import ABC
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from kfc_procedure.core.factory import BaseFactory


class BaseLocalModel(ABC, BaseEstimator):
    """
    Base class for local models used by the F-step.

    Local models are fitted separately for each cluster partition and are
    expected to expose the standard scikit-learn estimator interface.
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
