"""
FStep:
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator

from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.ml.base import LocalModelClassifierFactory, LocalModelRegressorFactory


class BaseFStep(ABC, BaseEstimator):
    """
    Fstep
    """
    models_: Dict

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, clusters: Dict) -> "BaseFStep":
        """Fit one local model per (divergence, cluster) pair."""
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray, clusters: Dict) -> Dict:
        """Return predictions for X."""
        ...

class FStep(BaseFStep):
    
    def __init__(self, config: Dict):
        self.config = config
    
    def fit(self, X: np.ndarray, y: np.ndarray, clusters: Dict[str, np.ndarray]) -> "FStep":

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}.")
        
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got X={X.shape[0]}, y={y.shape[0]})."
            )
        
        if not clusters:
            raise ValueError("clusters must not be empty.")
        
        self.global_mean = np.mean(y)
        
        name = self.config.get("name")
        
        if name in LocalModelRegressorFactory.available():
            self.task = "regression"
        elif name in LocalModelClassifierFactory.available():
            self.task = "classification"
        else:
            raise ValueError(
                f"Unknown local model: {name!r}. "
                f"Available regression: {LocalModelRegressorFactory.available()} | "
                f"classification: {LocalModelClassifierFactory.available()}"
            )
        
        params = self.config.get("params", {})
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self.models_        = {}
        self.label_order_   = {}

        for db, labels in clusters.items():
            self.models_[db] = {}

            for k in np.unique(labels):
                mask = labels == k
                if np.any(mask):

                    if self.task == "regression":
                        model = LocalModelRegressorFactory.create(name, **params)
                    else:
                        model = LocalModelClassifierFactory.create(name, **params)
                    
                    model.fit(X[mask], y[mask])
                    self.models_[db][k] = model

            self.label_order_[db] = sorted(self.models_[db].keys())
        
        return self
    
    def predict(self, X: np.ndarray, clusters: Dict[str, np.ndarray]):
        check_is_fitted(self, "models_")

        outputs = {}

        for db in clusters:
            n = X.shape[0]
            label_values = self.label_order_[db]
            col_index = {label: idx for idx, label in enumerate(label_values)}
            mat = np.zeros((n, len(label_values)))

            for k, model in self.models_[db].items():
                pred = model.predict(X)
                mat[:, col_index[k]] = pred
            
            outputs[db] = mat
        return outputs