"""
Estimator module for COBRA framework.

This package provides a unified interface for all estimators used
within COBRA-style ensemble and aggregation systems.

It includes:
- A base estimator interface (BaseEstimator)
- A registry-based factory system (EstimatorFactory)
- Native implementations (e.g., MeanRegressor)
- A scikit-learn adapter layer (SklearnEstimator)
- Utilities for automatic registration of sklearn models

Design goals:
- Consistent fit/predict API across all estimators
- Support for both custom and external (sklearn) models
- Dynamic model discovery via factory pattern
"""

from __future__ import annotations

from .base import (
    BaseEstimator,
    EstimatorFactory,
)
from .mean_regressor import MeanRegressor
from .sklearn import (
    SklearnEstimator,
    register_all_sklearn_estimators,
)

# register estimator sklearn
register_all_sklearn_estimators(EstimatorFactory)

__all__ = [
    "BaseEstimator",
    "EstimatorFactory",
    "MeanRegressor",
    "SklearnEstimator",
    "register_all_sklearn_estimators",
]