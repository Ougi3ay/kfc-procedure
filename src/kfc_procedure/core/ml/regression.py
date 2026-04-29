"""
kfc_procedure.models.regression
--------------------------------
Regression local model wrappers, auto-registered with LocalModelFactory.

Registered names
----------------
"linear"        – OrdinaryLeastSquares
"ridge"         – Ridge regression
"lasso"         – Lasso regression
"decision_tree" – DecisionTreeRegressor
"random_forest" – RandomForestRegressor
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from kfc_procedure.core.ml.base import BaseLocalModelRegressor, LocalModelRegressorFactory

@LocalModelRegressorFactory.register("linear_regression", "linear", "ols")
class LinearRegression(BaseLocalModelRegressor):

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        self.model_ = LR().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

@LocalModelRegressorFactory.register("ridge", "ridge_regression", "rr")
class RidgeRegression(BaseLocalModelRegressor):
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        self.model_ = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

@LocalModelRegressorFactory.register("lasso", "lasso_regression")
class LassoRegression(BaseLocalModelRegressor):
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        self.model_ = Lasso(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

@LocalModelRegressorFactory.register("decision_tree", "decision_tree_regression", "dtr")
class DecisionTreeRegression(BaseLocalModelRegressor):
    """Decision tree regressor."""

    def __init__(self, max_depth: int | None = None, random_state: int | None = None):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegression":
        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth, random_state=self.random_state
        ).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

@LocalModelRegressorFactory.register("random_forest", "random_forest_regression", "rf")
class RandomForestRegression(BaseLocalModelRegressor):
    """Random forest regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegression":
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
