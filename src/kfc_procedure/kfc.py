from __future__ import annotations

from typing import Dict
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from kfc_procedure.core.steps.cstep import CStep
from kfc_procedure.core.steps.fstep import FStep
from kfc_procedure.core.steps.kstep import KStep


class KFCProcedure(BaseEstimator):
    """
    Core KFC pipeline (paper-faithful version).

    K-step → F-step → C-step
    """

    def __init__(
        self,
        divergences,
        local_model,
        combiner,
        divergences_params: Dict = None,
        local_model_params: Dict = None,
        combiner_params: Dict = None,
        task: str = "regression",
        n_clusters=3,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError(
                "task must be 'regression' or 'classification'"
            )

        self.task = task

        self.divergences = divergences
        self.local_model = local_model
        self.combiner = combiner

        self.divergences_params = divergences_params or {}
        self.local_model_params = local_model_params or {}
        self.combiner_params = combiner_params or {}

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):

        X = np.asarray(X)
        y = np.asarray(y)

        X_k, X_l, y_k, y_l = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None
        )

        self.kstep_ = KStep(
            divergences=self.divergences,
            divergences_params=self.divergences_params,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        self.kstep_.fit(X_k)

        # training cluster assignments
        clusters_k = self.kstep_.clusters_

        # aggregation cluster assignments
        clusters_l = self.kstep_.predict(X_l)

        self.fstep_ = FStep(
            local_model=self.local_model,
            local_model_params=self.local_model_params,
            task=self.task,
            random_state=self.random_state
        )

        self.fstep_.fit(X_k, y_k, clusters_k)
        # M × K prediction matrix
        P_l = self.fstep_.predict(X_l, clusters_l)
        print(f"P_l : {P_l.shape}, y_l : {y_l.shape}")

        self.cstep_ = CStep(
            combiner=self.combiner,
            combiner_params=self.combiner_params,
            task=self.task,
            random_state=self.random_state
        )

        self.cstep_.fit(P_l, y_l)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(
            self,
            ["kstep_", "fstep_", "cstep_"]
        )

        X = np.asarray(X)

        # assign clusters
        clusters = self.kstep_.predict(X)

        # M × K prediction matrix
        P = self.fstep_.predict(X, clusters)

        # consensus aggregation
        return self.cstep_.predict(P)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if self.task != "classification":
            raise AttributeError(
                "predict_proba only available for classification"
            )

        check_is_fitted(
            self,
            ["kstep_", "fstep_", "cstep_"]
        )

        X = np.asarray(X)

        clusters = self.kstep_.predict(X)

        P = self.fstep_.predict_proba(X, clusters)

        return self.cstep_.predict_proba(P)

class KFCRegressor(KFCProcedure):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            task="regression",
            **kwargs
        )


class KFCClassifier(KFCProcedure):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            task="classification",
            **kwargs
        )