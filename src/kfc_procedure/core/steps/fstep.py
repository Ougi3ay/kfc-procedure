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
    def __init__(
        self,
        local_model: Union[str, BaseLocalModel],
        local_model_params: Dict = {},
        task: str = "regression",
    ):
        self.local_model = local_model
        self.local_model_params = local_model_params or {}
        self.task = task

    def fit(self, X, y, clusters: Dict[str, np.ndarray]):
        X = np.asarray(X)
        y = np.asarray(y)

        self.models_ = {}

        for div_name, cluster_ids in clusters.items():
            self.models_[div_name] = {}
            for k in np.unique(cluster_ids):
                idx = cluster_ids == k
                if np.sum(idx) == 0:
                    continue
                Xc, yc = X[idx], y[idx]
                model = self._resolve()
                model.fit(Xc, yc)

                self.models_[div_name][f"m{k}"] = {
                    "divergence" : div_name,
                    "cluster": int(k),
                    "model": model,
                }
        
        return self
    
    def predict(self, X, clusters: Dict[str, np.ndarray]):
        check_is_fitted(self, "models_")
        X = np.asarray(X)

        outputs = []
        for div_name, models in self.models_.items():
            pred = np.zeros(X.shape[0])
            cluster_ids = clusters[div_name]

            for _, meta in models.items():
                k = meta["cluster"]
                model = meta["model"]

                idx = cluster_ids == k

                if np.any(idx):
                    pred[idx] = model.predict(X[idx])
            outputs.append(pred.reshape(-1, 1))

        return np.column_stack(outputs)


    
    def _resolve(self) -> BaseLocalModel:
        """
        Build model from factory or reuse instance.
        """

        if not isinstance(self.local_model, str):
            return self.local_model

        name = self.local_model.lower()

        if not LocalModelFactory.contains(name):
            raise ValueError(
                f"Invalid local model: {name}. "
                f"Available: {LocalModelFactory.available()}"
            )

        if not LocalModelFactory.supports(name, self.task):
            raise ValueError(
                f"{name} not supported for task={self.task}. "
                f"Available: {LocalModelFactory.available_by_category(self.task)}"
            )

        return LocalModelFactory.create(
            name,
            **self.local_model_params
        )