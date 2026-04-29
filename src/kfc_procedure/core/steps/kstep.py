"""
KStep:
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.utils.resolve import resolve_kstep
from kfc_procedure.core.clustering.bregman import BregmanKMeans




class BaseKStep(ABC, BaseEstimator, ClusterMixin):
    clusters_: Dict

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> "BaseKStep":
        """Fit the clustering model on X."""
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> Dict:
        """Assign each sample to a cluster"""
        ...

class KStep(BaseKStep):
    def __init__(
        self,
        divergences: List[Union[str, Dict[str, Any], BregmanKMeans]]
    ):
        self.divergences = divergences
    
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KStep":
        """K-step: Fit one BregmanKMeans per divergence config on X."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        
        self.n_features_in_ : int                       = X.shape[1]
        self.models_        : Dict[str, BregmanKMeans]  = {}
        self.clusters_      : Dict[str, np.ndarray]     = {}

        resolved = resolve_kstep(self.divergences)

        for key, model in resolved.items():
            model.fit(X)
            self.models_[key] = model
            self.clusters_[key] = model.labels_.copy()

        return self

    def predict(self, X: np.ndarray):
        """Assign new samples to clusters for every divergence."""
        check_is_fitted(self, "models_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        
        return {
            key: model.predict(X)
            for key, model in self.models_.items()
        }
    
    def get_labels(self, key: str) -> np.ndarray:
        check_is_fitted(self, "clusters_")
        if key not in self.clusters_:
            raise ValueError(f"Invalid divergence key: {key!r}.")
        return self.clusters_[key]
    
    def get_centers(self, key: str) -> np.ndarray:
        check_is_fitted(self, "models_")
        if key not in self.models_:
            raise ValueError(f"Invalid divergence key: {key!r}.")
        return self.models_[key].cluster_centers_