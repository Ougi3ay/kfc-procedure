"""
K-step clustering stage for the KFC pipeline.

The K-step fits one BregmanKMeans model per divergence configuration and
tracks cluster assignments for each divergence variant.
"""
from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.clustering.divergences.base import BaseBregmanDivergence, BregmanDivergenceFactory
from kfc_procedure.core.clustering.bregman import BregmanKMeans

class KStep(ABC, BaseEstimator, ClusterMixin):
    def __init__(
        self,
        divergences: List[Union[str, BaseBregmanDivergence]],
        divergences_params: Dict = {},
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.divergences = divergences
        self.divergences_params = divergences_params
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)

        self.models_ = {}
        self.clusters_ = {}

        for div in self.divergences:
            name = self._get_name(div)
            params = self.divergences_params.get(name, {})
            divergence = self._resolve(div, params)

            model = BregmanKMeans(
                divergence=divergence,
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

            model.fit(X)

            self.models_[name] = model
            self.clusters_[name] = model.labels_
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        check_is_fitted(self, "models_")
        X = np.asarray(X, dtype=float)
        return {
            name: model.predict(X)
            for name, model in self.models_.items()
        }

    def _get_name(self, div):
        if isinstance(div, str):
            return div.lower()
        return getattr(div, "name", div.__class__.__name__).lower()

    def _resolve(self, div, params):
        if isinstance(div, str):
            return BregmanDivergenceFactory.create(div, **params)
        return div
