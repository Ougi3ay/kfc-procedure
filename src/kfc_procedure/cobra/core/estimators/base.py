"""
Base estimator interface and factory for COBRA-style learning systems.

This module defines the abstract contract for all estimators used
within the COBRA framework, along with a registry-based factory
for dynamic model instantiation.

Estimators in this framework are responsible for:

- Learning a mapping from input features to target outputs
- Producing predictions used as candidate signals in aggregation
  or ensemble procedures

The design follows a scikit-learn–like API, enforcing a consistent
`fit -> predict` workflow across all implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from kfc_procedure.cobra.core.factory import BaseFactory


class BaseEstimator(ABC):
    """
    Abstract base class for all COBRA estimators.

    This interface enforces a unified structure for supervised
    learning models that can be used within ensemble or aggregation
    pipelines.

    All estimators must implement:

    - fit: Train the model using input data
    - predict: Generate predictions for unseen samples

    Methods
    -------
    fit(x, y, **kwargs)
        Train the estimator on input features and targets.

    predict(x, **kwargs)
        Generate predictions for input features.

    Notes
    -----
    This design is intentionally compatible with scikit-learn-style
    estimators to simplify integration with external tools.

    Examples
    --------
    >>> class MyEstimator(BaseEstimator):
    ...     def fit(self, x, y):
    ...         return self
    ...
    ...     def predict(self, x):
    ...         return np.zeros(len(x))
    """

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "BaseEstimator":
        """
        Fit the estimator to training data.

        Parameters
        ----------
        x : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        y : np.ndarray
            Target values of shape (n_samples,).

        **kwargs : dict
            Additional implementation-specific arguments.

        Returns
        -------
        BaseEstimator
            Fitted estimator instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for input samples.

        Parameters
        ----------
        x : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        **kwargs : dict
            Additional implementation-specific arguments.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        raise NotImplementedError


class EstimatorFactory(BaseFactory):
    """
    Registry-based factory for estimator classes.

    This factory enables dynamic registration and creation of
    estimators using string identifiers. It is used to decouple
    model selection from implementation details.

    Notes
    -----
    Each estimator registered under this factory can be instantiated
    using:

    - EstimatorFactory.create(name, **kwargs)

    Examples
    --------
    >>> @EstimatorFactory.register("linear")
    ... class LinearEstimator(BaseEstimator):
    ...     pass

    >>> model = EstimatorFactory.create("linear")
    """

    pass
