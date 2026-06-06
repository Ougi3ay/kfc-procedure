"""
Mean Regressor implementation for regression tasks.
"""
from __future__ import annotations
import numpy as np

from kfc_procedure.cobra.core.estimators.base import BaseEstimator
from kfc_procedure.cobra.core.estimators.base import EstimatorFactory


@EstimatorFactory.register("mean_regressor", "mean")
class MeanRegressor(BaseEstimator):
    """
    Mean baseline regressor.

    This estimator ignores input features and always predicts
    the mean of the training target values.

    It is commonly used as a sanity-check baseline in regression
    problems to ensure that more complex models provide meaningful
    improvement.

    Attributes
    ----------
    mean_ : float
        Mean value of training targets.

    Methods
    -------
    fit(x, y)
        Computes and stores the mean of y.

    predict(x)
        Returns a constant prediction equal to the training mean.
    """

    def __init__(self) -> None:
        self.mean_ = None

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "MeanRegressor":
        """
        Fit the model by computing the mean of y.

        Parameters
        ----------
        x : np.ndarray
            Ignored input features.

        y : np.ndarray
            Target values.

        Returns
        -------
        MeanRegressor
            Fitted instance.
        """
        y = np.asarray(y)

        if y.ndim == 0:
            raise ValueError("y must be a 1D array-like structure")

        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict constant mean value for all inputs.

        Parameters
        ----------
        x : np.ndarray
            Input features (ignored).

        Returns
        -------
        np.ndarray
            Constant predictions equal to training mean.
        """
        if self.mean_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        x = np.asarray(x)
        return np.full(shape=(len(x),), fill_value=self.mean_, dtype=float)
