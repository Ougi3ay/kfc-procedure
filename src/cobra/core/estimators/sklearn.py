"""
Create
"""
import inspect

from sklearn.base import (
    BaseEstimator as SkBaseEstimator,
    ClassifierMixin,
    RegressorMixin
)
from sklearn.utils import all_estimators
from cobra.core.estimators.base import BaseEstimator
from cobra.utils.preprocessing import clean_sklearn_name


class SklearnEstimator(BaseEstimator):

    def __init__(self, estimator_cls: type[SkBaseEstimator], **kwargs):
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
            f"{self.estimator.__class__.__name__} has no predict_proba"
        )
    
    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

def register_all_sklearn_estimators(factory):
    """
    Auto-register all sklearn estimators into COBRA factory.
    """

    for name, cls in all_estimators():

        try:
            # 1. must be class
            if not inspect.isclass(cls):
                continue

            # 2. must support fit
            if not hasattr(cls, "fit"):
                continue

            # 3. only ML models (skip transformers, preprocessors, etc.)
            if not (
                issubclass(cls, ClassifierMixin)
                or issubclass(cls, RegressorMixin)
            ):
                continue

            # 4. normalize name
            key = clean_sklearn_name(name)

            # 5. safe builder (avoid late binding issues)
            def builder(estimator_cls=cls, **kwargs):
                return SklearnEstimator(estimator_cls, **kwargs)

            # 6. register properly
            factory.register(key)(builder)

        except Exception:
            continue
