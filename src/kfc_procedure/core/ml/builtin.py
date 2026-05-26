from __future__ import annotations

import inspect
import re
from typing import Type

import numpy as np
from numpy.typing import ArrayLike

from sklearn.dummy import DummyRegressor
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator

from kfc_procedure.core.ml.base import BaseLocalModel, LocalModelFactory


@LocalModelFactory.register(
    "mean_regressor",
    "dummy_mean",
    categories={"regression"},
)
class MeanRegressor(BaseLocalModel):
    def __init__(self) -> None:
        self.estimator = DummyRegressor(strategy="mean")

    def fit(self, X: ArrayLike, y: ArrayLike):
        self.estimator.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


class SklearnLocalModel(BaseLocalModel):
    """
    Adapter wrapping sklearn estimators into F-step local models.
    """

    def __init__(self, model_cls: Type[BaseEstimator], **kwargs):
        self.model = model_cls(**kwargs)

        # store capability flags
        self._has_proba = hasattr(self.model, "predict_proba")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self._has_proba:
            return self.model.predict_proba(X)
        return None

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


def clean_sklearn_name(name: str) -> str:
    """
    Convert sklearn estimator name to snake_case.
    """

    if name.isupper():
        return name.lower()

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)

    return s2.lower()


def register_all_sklearn_models():
    """
    Auto-register sklearn models into LocalModelFactory.
    """

    seen_keys = set()

    for name, cls in all_estimators():
        try:
            if not inspect.isclass(cls):
                continue

            if not hasattr(cls, "fit"):
                continue

            is_classifier = issubclass(cls, ClassifierMixin)
            is_regressor = issubclass(cls, RegressorMixin)

            if not (is_classifier or is_regressor):
                continue

            key = clean_sklearn_name(name)

            # avoid collisions
            if key in seen_keys:
                key = f"{key}_{cls.__name__.lower()}"

            seen_keys.add(key)

            category = (
                "classification"
                if is_classifier
                else "regression"
            )

            def builder(model_cls=cls, **kwargs):
                return SklearnLocalModel(model_cls, **kwargs)

            LocalModelFactory.register(
                key,
                categories={category}
            )(builder)

        except Exception:
            continue
