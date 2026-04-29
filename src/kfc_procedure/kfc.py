
from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_is_fitted

from kfc_procedure.base import BaseKFC


class KFCRegressor(BaseKFC):
    """
    KFC meta-estimator for regression.

    This estimator fits the full KFC pipeline and returns continuous
    predictions produced by the fitted C-step aggregator.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for X."""
        return self.cstep_.predict(self._predict_internal(X))


class KFCClassifier(BaseKFC):
    """
    KFC meta-estimator for classification.

    This estimator uses a stratified internal split during fit and exposes
    both label prediction and probability prediction when supported by the
    configured C-step strategy.
    """
     
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "KFCClassifier":
        """
        Fit the classifier using stratified sampling by label.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : np.ndarray
            Class labels of shape (n_samples,).

        Returns
        -------
        KFCClassifier
            Fitted classifier.
        """
        return super().fit(X, y, stratify=y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return self.cstep_.predict(self._predict_internal(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Raises
        ------
        AttributeError
            If the configured C-step aggregator does not support probability
            prediction.
        """
        check_is_fitted(self, "is_fitted_")
        if not hasattr(self.cstep_, "predict_proba"):
            raise AttributeError(
                f"{type(self.cstep_).__name__} does not implement predict_proba."
            )
        return self.cstep_.predict_proba(self._predict_internal(X))
    