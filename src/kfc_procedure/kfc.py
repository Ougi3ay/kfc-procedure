
from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.base import BaseKFC


class KFCRegressor(BaseKFC):
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return continuous predictions for X."""
        return self.cstep_.predict(self._predict_internal(X))
class KFCClassifier(BaseKFC):
     
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "KFCClassifier":
        """Fit with stratify=y to keep class proportions"""
        return super().fit(X, y, stratify=y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.cstep_.predict(self._predict_internal(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        if not hasattr(self.cstep_, "predict_proba"):
            raise AttributeError(
                f"{type(self.cstep_).__name__} does not implement predict_proba."
            )
        return self.cstep_.predict_proba(self._predict_internal(X))
    