"""
Local model layer for F-step in the KFC pipeline.

Local models are trained on cluster/divergence-specific data and produce
a prediction matrix consumed by the C-step combiner.

This design is task-agnostic at the base class level and uses a
registry-based factory for task separation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from kfc_procedure.core.factory import BaseFactory

class BaseLocalModel(BaseEstimator, ABC):
    """
    Unified base class for all local models in the F-step.

    A local model learns from a subset of data (e.g. cluster-wise or
    divergence-specific partition) and produces predictions that are
    later combined in the C-step.

    This class is intentionally task-agnostic.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the local model.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        y : np.ndarray
            Target values.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for F-step.

        Returns predictions that will be stacked into a prediction
        matrix for the C-step.
        """
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LocalModelFactory(BaseFactory):
    """
    Factory for all local models used in F-step.

    Supports both regression and classification models via categories:

    - regression
    - classification
    - multitask (optional extension)
    """
    ...