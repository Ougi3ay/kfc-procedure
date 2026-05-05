
from __future__ import annotations
from abc import ABC
from typing import Dict
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from kfc_procedure.core.steps.cstep import CStep
from kfc_procedure.core.steps.fstep import FStep
from kfc_procedure.core.steps.kstep import KStep


class KFCProcedure(ABC, BaseEstimator):
    """Core KFC pipeline.

    The KFCProcedure runs the full KFC pipeline by executing the K-step,
    F-step, and C-step in sequence.

    Parameters
    ----------
    divergences : list
        Divergence identifiers or divergence objects.
    local_model : str or BaseLocalModelRegressor or BaseLocalModelClassifier
        Local model identifier or instance.
    aggregation : str or BaseAggregationRegressor or BaseAggregationClassifier
        Aggregation strategy identifier or instance.
    divergences_param : dict, optional
        Parameters for each divergence.
    local_model_param : dict, optional
        Parameters for the local model factory.
    aggregation_param : dict, optional
        Parameters for the aggregation strategy.
    task : str, default="regression"
        Either ``"regression"`` or ``"classification"``.
    n_clusters : int, default=8
        Number of clusters for each divergence.
    max_iter : int, default=300
        Maximum number of clustering iterations.
    tol : float, default=1e-4
        Clustering tolerance for convergence.
    verbose : bool, default=False
        Whether to show verbose output during clustering.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        divergences,
        local_model,
        aggregation,
        divergences_param: Dict = None,
        local_model_param: Dict = None,
        aggregation_param: Dict = None,
        task: str = "regression",
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")

        self.task = task
        self.divergences = divergences
        self.local_model = local_model
        self.aggregation = aggregation

        self.divergences_param = divergences_param or {}
        self.local_model_param = local_model_param or {}
        self.aggregation_param = aggregation_param or {}

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the full KFC pipeline.

        Parameters
        ----------
        X : ndarray
            Training features.
        y : ndarray
            Target values.

        Returns
        -------
        self : KFCProcedure
            The fitted KFC procedure.
        """
        self.kstep_ = KStep(
            self.divergences,
            self.divergences_param,
            self.n_clusters,
            self.max_iter,
            self.tol,
            self.verbose,
            self.random_state,
        ).fit(X)

        clusters = self.kstep_.clusters_

        self.fstep_ = FStep(
            self.local_model,
            self.local_model_param,
            self.task,
        ).fit(X, y, clusters)

        preds = self.fstep_.predict(X, clusters)
        X_c = np.hstack(list(preds.values()))

        self.cstep_ = CStep(
            self.aggregation,
            self.aggregation_param,
            self.task,
        ).fit(X_c, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the fitted KFC pipeline.

        Parameters
        ----------
        X : ndarray
            Input features.

        Returns
        -------
        ndarray
            Final aggregated predictions.
        """
        check_is_fitted(self, ["kstep_", "fstep_", "cstep_"])

        clusters = self.kstep_.predict(X)
        preds = self.fstep_.predict(X, clusters)
        X_c = np.hstack(list(preds.values()))

        return self.cstep_.predict(X_c)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for classification.

        Parameters
        ----------
        X : ndarray
            Input features.

        Returns
        -------
        ndarray
            Aggregated class probabilities.

        Raises
        ------
        AttributeError
            If the configured task is not classification.
        """
        if self.task != "classification":
            raise AttributeError("Only available for classification")

        clusters = self.kstep_.predict(X)
        probas = self.fstep_.predict_proba(X, clusters)
        X_c = np.hstack(list(probas.values()))

        return self.cstep_.predict_proba(X_c)


class KFCRegressor(KFCProcedure):
    """KFC for regression."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="regression", **kwargs)


class KFCClassifier(KFCProcedure):
    """KFC for classification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task="classification", **kwargs)
