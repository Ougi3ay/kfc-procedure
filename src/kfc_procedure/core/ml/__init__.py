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
from kfc_procedure.core.lm.base import BaseLocalModel
from kfc_procedure.core.lm.regression import (      # noqa: F401 – triggers registration
    DecisionTreeRegression,
    LassoRegression,
    LinearRegression,
    RandomForestRegression,
    RidgeRegression,
)
from kfc_procedure.core.lm.classification import (  # noqa: F401 – triggers registration
    DecisionTreeClassifier,
    LogisticClassifier,
    RandomForestClassifier,
)

__all__ = [
    "BaseLocalModel",
    # Regression
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "DecisionTreeRegression",
    "RandomForestRegression",
    # Classification
    "LogisticClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
]