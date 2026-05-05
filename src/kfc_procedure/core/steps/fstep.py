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
    BaseLocalModelClassifier,
    BaseLocalModelRegressor,
    LocalModelClassifierFactory,
    LocalModelRegressorFactory
)

class FStep(ABC, BaseEstimator):
    """Local model learning step.

    The F-step fits one local model for each divergence/cluster pair.

    Parameters
    ----------
    local_model : str or BaseLocalModelRegressor or BaseLocalModelClassifier
        Local model identifier or instance to train on each cluster.
    local_model_param : dict
        Parameters passed to the local model factory when building models.
    task : str
        Either ``"regression"`` or ``"classification"``.

    Attributes
    ----------
    models_ : dict
        Nested dictionary of fitted local models indexed by divergence name
        and cluster label.
    """

    def __init__(
        self,
        local_model: Union[str, BaseLocalModelRegressor, BaseLocalModelClassifier],
        local_model_param: Dict,
        task: str,
    ):
        self.local_model = local_model
        self.local_model_param = local_model_param
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, clusters: Dict[str, np.ndarray]):
        """Fit local models for each divergence/cluster combination.

        Parameters
        ----------
        X : ndarray
            Training features.
        y : ndarray
            Training targets.
        clusters : dict
            Cluster labels for each divergence, keyed by divergence name.

        Returns
        -------
        self : FStep
            The fitted F-step instance.
        """
        self.models_ = {}

        for div_name, cluster_ids in clusters.items():
            self.models_[div_name] = {}

            for k in np.unique(cluster_ids):
                idx = cluster_ids == k
                model = self._build_model()
                model.fit(X[idx], y[idx])
                self.models_[div_name][k] = model

        return self

    def predict(self, X: np.ndarray, clusters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict using local models for each divergence.

        Parameters
        ----------
        X : ndarray
            Input features to predict.
        clusters : dict
            Cluster labels for each divergence.

        Returns
        -------
        dict
            Predictions for each divergence, with shape (n_samples, 1).
        """
        check_is_fitted(self, "models_")
        predictions = {}

        for div_name, cluster_ids in clusters.items():
            preds = np.zeros(X.shape[0])

            for k, model in self.models_[div_name].items():
                idx = cluster_ids == k
                if np.any(idx):
                    preds[idx] = model.predict(X[idx])

            predictions[div_name] = preds.reshape(-1, 1)

        return predictions

    def predict_proba(self, X: np.ndarray, clusters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict class probabilities for classification local models.

        Parameters
        ----------
        X : ndarray
            Input features to predict.
        clusters : dict
            Cluster labels for each divergence.

        Returns
        -------
        dict
            Class probability arrays for each divergence.

        Raises
        ------
        AttributeError
            If the task is not classification.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba only available for classification")

        probas = {}

        for div_name, cluster_ids in clusters.items():
            probs = []

            for k, model in self.models_[div_name].items():
                idx = cluster_ids == k
                if np.any(idx) and hasattr(model, "predict_proba"):
                    probs.append(model.predict_proba(X[idx]))

            if probs:
                probas[div_name] = np.vstack(probs)

        return probas

    def _build_model(self):
        """
        Build a local model based on task type and configuration.

        Returns
        -------
        BaseEstimator
            Instantiated local model.

        Raises
        ------
        ValueError
            If the model name is not found in the corresponding factory.
        """
        if not isinstance(self.local_model, str):
            return self.local_model

        name = self.local_model

        if self.task == "regression":
            if name in LocalModelRegressorFactory.available():
                return LocalModelRegressorFactory.create(
                    name, **self.local_model_param
                )

            raise ValueError(
                f"'{name}' is not a valid REGRESSION model. "
                f"Available: {LocalModelRegressorFactory.available()}"
            )

        if self.task == "classification":
            if name in LocalModelClassifierFactory.available():
                return LocalModelClassifierFactory.create(
                    name, **self.local_model_param
                )

            raise ValueError(
                f"'{name}' is not a valid CLASSIFICATION model. "
                f"Available: {LocalModelClassifierFactory.available()}"
            )

        raise ValueError(f"Unknown task '{self.task}'")
