"""
Sklearn estimator adapter for COBRA framework.

This module provides a bridge between scikit-learn estimators
and the COBRA estimator interface. It enables automatic wrapping
and registration of sklearn-compatible models into the COBRA
factory system.

Key features:
- Uniform interface (fit / predict / predict_proba)
- Dynamic wrapping of sklearn estimators
- Automatic registration into COBRA factory
"""

import inspect
from typing import Type

from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators

from cobra.core.estimators.base import BaseEstimator
from cobra.utils.preprocessing import clean_sklearn_name

class SklearnEstimator(BaseEstimator):
    """
    Wrapper for sklearn estimators to make them compatible
    with the COBRA BaseEstimator interface.
    """

    def __init__(self, estimator_cls: Type[SkBaseEstimator], **kwargs):
        self.estimator = estimator_cls(**kwargs)

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x)

        raise NotImplementedError(
            f"{self.estimator.__class__.__name__} does not support predict_proba"
        )

    def get_params(self, deep: bool = True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

def register_all_sklearn_estimators(factory):
    """
    Register all compatible sklearn estimators into COBRA factory.

    Only estimators that:
    - implement fit
    - are classifiers or regressors
    are included.

    Parameters
    ----------
    factory : BaseFactory
        COBRA estimator factory instance.

    Notes
    -----
    This function is intended for dynamic plugin-style registration
    of sklearn models into the COBRA ecosystem.
    """

    for name, cls in all_estimators():

        try:
            # Skip invalid entries
            if not inspect.isclass(cls):
                continue

            # Must support fit
            if not hasattr(cls, "fit"):
                continue

            # Only ML estimators (not transformers, etc.)
            if not (
                issubclass(cls, ClassifierMixin)
                or issubclass(cls, RegressorMixin)
            ):
                continue

            key = clean_sklearn_name(name)

            # Avoid late-binding bug
            def builder(estimator_cls=cls, **kwargs):
                return SklearnEstimator(estimator_cls, **kwargs)

            factory.register(key)(builder)

        except Exception:
            continue
