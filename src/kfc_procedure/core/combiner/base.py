"""
Base combiner interfaces and factory infrastructure.

This module defines the abstract BaseCombiner and a registry-based
CombinerFactory used to dynamically select ensemble combination
strategies for regression and classification.

The C-step combines divergence-specific predictions into a final
ensemble prediction.

Supported strategies
--------------------

Regression:
    "mean"          – row-wise mean (stateless)
    "weighted_mean" – OLS-learned weighted average
    "stacking"      – meta-regressor trained on prediction matrix

Classification:
    "majority_vote" – row-wise mode across base predictions
    "stacking"      – meta-classifier trained on prediction matrix
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator

from kfc_procedure.core.factory import BaseFactory

class BaseCombiner(ABC, BaseEstimator):
    """
    Abstract base class for ensemble combination strategies.

    A combiner takes a prediction matrix:

        shape = (n_samples, n_models)

    and produces a final aggregated prediction.

    This follows the scikit-learn estimator interface.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the combiner (if required).

        Parameters
        ----------
        X : np.ndarray
            Prediction matrix from base models
            shape = (n_samples, n_models)

        y : np.ndarray, optional
            True targets (required for supervised combiners
            like stacking or weighted mean)

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, X: np.ndarray) -> np.ndarray:
        """
        Combine predictions into final output.

        Parameters
        ----------
        X : np.ndarray
            Prediction matrix from base models

        Returns
        -------
        np.ndarray
            Final combined prediction
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scikit-learn compatible prediction interface.
        """
        return self.combine(X)

class CombinerFactory(BaseFactory):
    ...
