"""
K-step clustering stage for KFCProcedure.

This module implements the K-step of the KFC pipeline, where multiple
Bregman-KMeans models are trained independently under different
divergence assumptions.

Each divergence induces a different geometric view of the data space,
resulting in multiple clustering partitions of the same dataset.

Overview
--------
Given a set of divergences:

    D = {d_1, d_2, ..., d_m}

the K-step fits one BregmanKMeans model per divergence:

    model_i = KMeans(d_i, K)

This produces a collection of clustering assignments:

    C_i = model_i.predict(X)

which are stored for downstream fusion or selection stages.

Key Idea
--------
Instead of committing to a single metric space, the K-step maintains
multiple clustering hypotheses induced by different Bregman divergences.
This enables robust clustering under heterogeneous data distributions.

Supported divergences
---------------------
Divergences can be provided either as:

* string identifiers (resolved via BregmanDivergenceFactory)
* instantiated BaseBregmanDivergence objects

Examples include:
* squared Euclidean
* generalized KL divergence
* Itakura–Saito divergence
* logistic divergence

Outputs
-------
models_
    Dictionary mapping divergence name → fitted BregmanKMeans model.

clusters_
    Dictionary mapping divergence name → training cluster labels.

Methods
-------
fit(X)
    Fit one clustering model per divergence.

predict(X)
    Return cluster assignments for each divergence model.

Notes
-----
This stage does not enforce consensus across divergences.
Each model is trained independently and stored separately.

The resulting structure is intended for ensemble clustering or
subsequent aggregation steps in the KFC pipeline.
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
    """
    Multi-divergence clustering stage in the KFC pipeline.

    This estimator fits multiple BregmanKMeans models, each using a
    different Bregman divergence. The goal is to produce multiple
    clustering representations of the same dataset under different
    geometric assumptions.

    Parameters
    ----------
    divergences : list of str or BaseBregmanDivergence
        List of divergence specifications. Each element can be:
        - a string identifier resolved via BregmanDivergenceFactory
        - an instantiated divergence object

    divergences_params : dict, default={}
        Optional parameter dictionary per divergence name.

        Example:
            {
                "gkl": {"alpha": 1.0},
                "is": {"scale": 0.5}
            }

    n_clusters : int, default=3
        Number of clusters per divergence model.

    max_iter : int, default=300
        Maximum number of Lloyd iterations per KMeans model.

    tol : float, default=1e-4
        Convergence tolerance for distortion change.

    verbose : bool, default=False
        If True, prints convergence diagnostics.

    random_state : int or None, default=None
        Random seed for reproducibility across all models.

    Attributes
    ----------
    models_ : dict
        Fitted BregmanKMeans models keyed by divergence name.

    clusters_ : dict
        Training cluster assignments per divergence model.

    Methods
    -------
    fit(X, y=None)
        Fit one clustering model per divergence.

    predict(X)
        Return cluster assignments for each divergence model.

    Notes
    -----
    This stage is purely model-parallel:

    * No divergence interaction occurs during training
    * Each model is independent
    * Outputs are intended for downstream ensemble fusion

    The design supports heterogeneous metric learning where no single
    divergence is assumed optimal for the dataset structure.
    """
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
