"""
Estimator wrappers for COBRA-style expert pools.

This module provides a unified estimator layer used in COBRA-based
ensemble systems such as GradientCOBRA and MixCOBRA.

It standardizes scikit-learn models under a single interface so they
can be:

- registered via a factory
- swapped dynamically in pipelines
- used consistently in ensemble expert pools

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Design goals
------------
- unify estimator interface across models
- support factory-based instantiation
- reduce boilerplate for sklearn models
- enable hyperparameter injection via constructors
- ensure compatibility with ensemble aggregation systems
"""

from __future__ import annotations

from typing import Optional
from numpy.typing import ArrayLike

import numpy as np

from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .base import BaseEstimator, EstimatorFactory

@EstimatorFactory.register("mean_regressor", "dummy_mean")
class MeanRegressor(BaseEstimator):
    """
    Mean baseline regressor.
    """

    def __init__(self) -> None:
        self.estimator = DummyRegressor(strategy="mean")

    def fit(self, x: ArrayLike, y: ArrayLike):
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(self.estimator.predict(x), dtype=float)
