from __future__ import annotations

from typing import Dict
import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from kfc_procedure.core.steps.cstep import CStep
from kfc_procedure.core.steps.fstep import FStep
from kfc_procedure.core.steps.kstep import KStep


class KFCProcedure(BaseEstimator):
    """
    Core KFC pipeline.

    Runs:
        K-step → F-step → C-step
    """

    def __init__(
        self,
        divergences,
        local_model,
        combiner,
        divergences_param: Dict = None,
        local_model_param: Dict = None,
        combiner_param: Dict = None,
        task: str = "regression",
        n_clusters=3,
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
        self.combiner = combiner

        self.divergences_param = divergences_param or {}
        self.local_model_param = local_model_param or {}
        self.combiner_param = combiner_param or {}

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):

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
            self.combiner,
            self.combiner_param,
            self.task,
        ).fit(X_c, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(self, ["kstep_", "fstep_", "cstep_"])

        clusters = self.kstep_.predict(X)
        preds = self.fstep_.predict(X, clusters)
        X_c = np.hstack(list(preds.values()))

        return self.cstep_.predict(X_c)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

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