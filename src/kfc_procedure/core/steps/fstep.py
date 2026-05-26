"""
F-step local fitting stage for the KFC pipeline.

The F-step fits one local model for each combination of divergence and
cluster assignment returned by the preceding K-step.
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.core.ml.base import (
    BaseLocalModel,
    LocalModelFactory,
)


class FStep(ABC, BaseEstimator):
    """
    F-step: trains local models per divergence-cluster pair.

    This version uses a unified LocalModelFactory instead of separate
    regression/classification factories.
    """

    def __init__(
        self,
        local_model: Union[str, BaseLocalModel],
        local_model_param: Dict,
        task: str,
    ):
        self.local_model = local_model
        self.local_model_param = local_model_param or {}
        self.task = task

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, X: np.ndarray, y: np.ndarray, clusters: Dict[str, np.ndarray]):
        """
        Fit local models for each divergence/cluster combination.
        """

        self.models_: Dict[str, Dict[int, BaseLocalModel]] = {}

        for div_name, cluster_ids in clusters.items():
            self.models_[div_name] = {}

            for k in np.unique(cluster_ids):
                idx = cluster_ids == k

                model = self._build_model()

                model.fit(X[idx], y[idx])

                self.models_[div_name][k] = model

        return self

    # =========================================================
    # PREDICT
    # =========================================================
    def predict(
        self,
        X: np.ndarray,
        clusters: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:

        check_is_fitted(self, "models_")

        predictions = {}

        for div_name, cluster_ids in clusters.items():

            preds = np.zeros(X.shape[0], dtype=float)

            for k, model in self.models_[div_name].items():
                idx = cluster_ids == k

                if np.any(idx):
                    preds[idx] = model.predict(X[idx])

            predictions[div_name] = preds.reshape(-1, 1)

        return predictions

    # =========================================================
    # PROBABILITIES (classification only)
    # =========================================================
    def predict_proba(
        self,
        X: np.ndarray,
        clusters: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:

        if self.task != "classification":
            raise AttributeError("predict_proba only available for classification")

        check_is_fitted(self, "models_")

        probas = {}

        for div_name, cluster_ids in clusters.items():

            n_samples = X.shape[0]
            combined = None

            for k, model in self.models_[div_name].items():
                idx = cluster_ids == k

                if not np.any(idx):
                    continue

                if not hasattr(model, "predict_proba"):
                    continue

                p = model.predict_proba(X[idx])

                if combined is None:
                    combined = np.zeros((n_samples, p.shape[1]))

                combined[np.where(idx)[0]] = p

            if combined is not None:
                probas[div_name] = combined

        return probas

    # =========================================================
    # MODEL BUILDER (NEW UNIFIED FACTORY)
    # =========================================================
    def _build_model(self) -> BaseLocalModel:
        """
        Build a local model using LocalModelFactory.
        """

        # already an instance → reuse directly
        if not isinstance(self.local_model, str):
            return self.local_model

        name = self.local_model.lower()

        # validate existence
        if not LocalModelFactory.contains(name):
            raise ValueError(
                f"'{name}' is not a valid local model. "
                f"Available: {LocalModelFactory.available()}"
            )

        # optional task safety check
        if not LocalModelFactory.supports(name, self.task):
            raise ValueError(
                f"'{name}' is not valid for task='{self.task}'. "
                f"Available for '{self.task}': "
                f"{LocalModelFactory.available_by_category(self.task)}"
            )

        return LocalModelFactory.create(
            name,
            **self.local_model_param
        )