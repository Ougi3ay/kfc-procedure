"""
K-step clustering stage for the KFC pipeline.

The K-step fits one BregmanKMeans model per divergence configuration and
tracks cluster assignments for each divergence variant.
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
    """
    Abstract interface for the K-step clustering component.
    """

    clusters_: Dict

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> "BaseKStep":
        """
        Fit the clustering model on X.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : np.ndarray | None, default=None
            Optional target values.

        Returns
        -------
        BaseKStep
            Fitted clustering stage.
        """
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> Dict:
        """
        Assign each sample to a cluster.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dict
            Mapping from divergence key to cluster labels.
        """
        ...

class KStep(BaseKStep):
    """
    Concrete K-step implementation that resolves a list of divergence
    configurations into BregmanKMeans clustering models.

    Parameters
    ----------
    config : dict
    """

    def __init__(
        self,
        config: Dict,
    ):
        self.config = config
        self.divergences = config.get("divergences", [])
    
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KStep":
        """
        Fit one BregmanKMeans model per divergence configuration.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        y : np.ndarray | None, default=None
            Optional target labels for compatibility with scikit-learn.

        Returns
        -------
        KStep
            Fitted KStep instance.
        """
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
        """
        Assign new samples to clusters for every divergence.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dict[str, np.ndarray]
            Cluster label arrays keyed by divergence name.
        """
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
        """
        Return the cluster labels for a specific divergence key.

        Parameters
        ----------
        key : str
            Divergence identifier.

        Returns
        -------
        np.ndarray
            Cluster assignment labels for the requested divergence.
        """
        check_is_fitted(self, "clusters_")
        if key not in self.clusters_:
            raise ValueError(f"Invalid divergence key: {key!r}.")
        return self.clusters_[key]

    def get_centers(self, key: str) -> np.ndarray:
        """
        Return cluster centers for a specific divergence key.

        Parameters
        ----------
        key : str
            Divergence identifier.

        Returns
        -------
        np.ndarray
            Cluster centers for the requested divergence.
        """
        check_is_fitted(self, "models_")
        if key not in self.models_:
            raise ValueError(f"Invalid divergence key: {key!r}.")
        return self.models_[key].cluster_centers_