"""
kfc_procedure.models
--------------------
Local model wrappers used by LocalModelFStep.

Importing this package registers all built-in models with LocalModelFactory.

Regression models  (task="regression")
---------------------------------------
    "linear"        – OrdinaryLeastSquares
    "ridge"         – Ridge regression
    "lasso"         – Lasso regression
    "decision_tree" – DecisionTreeRegressor
    "random_forest" – RandomForestRegressor

Classification models  (task="classification")
-----------------------------------------------
    "logistic"      – LogisticRegression
    "decision_tree" – DecisionTreeClassifier
    "random_forest" – RandomForestClassifier
"""
from kfc_procedure.core.ml.base import BaseLocalModel, BaseLocalModelRegressor, BaseLocalModelClassifier
from kfc_procedure.core.ml.regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    DecisionTreeRegression,
    RandomForestRegressor,
)
from kfc_procedure.core.ml.classification import (
    LogisticClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
)

__all__ = [
    "BaseLocalModel",
    "BaseLocalModelRegressor",
    "BaseLocalModelClassifier",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "DecisionTreeRegression",
    "RandomForestRegressor",
    "LogisticClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
]