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
    """Clustering step of the KFC procedure.

    The K-step fits one clustering model per divergence and stores the
    cluster assignments used by the F-step.

    Parameters
    ----------
    divergences : list
        Divergence identifiers or divergence objects.
    divergences_param : dict
        Parameters for each divergence. The divergence name is used as key.
    n_clusters : int, default=3
        Number of clusters for each KMeans model.
    max_iter : int, default=300
        Maximum number of iterations for the clustering solver.
    tol : float, default=1e-4
        Convergence tolerance for the clustering solver.
    verbose : bool, default=False
        Whether to print progress messages during fitting.
    random_state : int or None
        Random seed for clustering initialization.

    Attributes
    ----------
    models_ : dict
        Fitted clustering models keyed by divergence name.
    clusters_ : dict
        Cluster assignments for each fitted divergence.
    """

    def __init__(
        self,
        divergences: List[Union[str, BaseBregmanDivergence]],
        divergences_param: Dict,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.divergences = divergences
        self.divergences_param = divergences_param
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit one clustering model per divergence.

        Parameters
        ----------
        X : ndarray
            Input feature matrix.
        y : ndarray or None
            Ignored. Included for sklearn compatibility.

        Returns
        -------
        self : KStep
            The fitted K-step instance.
        """
        X = np.asarray(X, dtype=float)

        self.models_ = {}
        self.clusters_ = {}

        for div in self.divergences:
            # divergence
            name = self._get_divergence_name(div)
            div_param = self.divergences_param.get(name, {})
            div_instance = self._resolve_divergence(div, div_param)

            model = BregmanKMeans(
                divergence=div_instance,
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            model.fit(X)

            self.models_[name] = model
            self.clusters_[name] = model.labels_

        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict cluster assignments for each divergence.

        Parameters
        ----------
        X : ndarray
            Input feature matrix.

        Returns
        -------
        dict
            Cluster labels keyed by divergence name.
        """
        check_is_fitted(self, "models_")
        X = np.asarray(X, dtype=float)

        return {name: model.predict(X) for name, model in self.models_.items()}
    
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
    
    def _get_divergence_name(self, div):
        if isinstance(div, str):
            return div.lower()
        return getattr(div, "name", div.__class__.__name__).lower()

    def _resolve_divergence(self, div, params):
        """Convert string or instance into a divergence instance."""
        if isinstance(div, str):
            return BregmanDivergenceFactory.create(div, **params)
        elif isinstance(div, BaseBregmanDivergence):
            return div
        else:
            raise TypeError(f"Invalid divergence: {div!r}")

