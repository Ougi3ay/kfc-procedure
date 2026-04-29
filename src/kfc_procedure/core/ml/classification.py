"""
kfc_procedure.models.classification
-------------------------------------
Classification local model wrappers, auto-registered with LocalModelFactory.

Registered names
----------------
"logistic"      – LogisticRegression
"decision_tree" – DecisionTreeClassifier
"random_forest" – RandomForestClassifier
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier

from kfc_procedure.core.ml.base import BaseLocalModelClassifier, LocalModelClassifierFactory



@LocalModelClassifierFactory.register("logistic", "logistic_regression", "lr")
class LogisticClassifier(BaseLocalModelClassifier):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int | None = None,
        **kwargs,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

@LocalModelClassifierFactory.register("decision_tree", "dt")
class DecisionTreeClassifier(BaseLocalModelClassifier):
    def __init__(self, max_depth: int | None = None, random_state: int | None = None):
        self.max_depth = max_depth
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.model_ = SkDecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state
        )
        self.model_.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

@LocalModelClassifierFactory.register("random_forest", "rf")
class RandomForestClassifier(BaseLocalModelClassifier):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        self.model_ = SkRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)
