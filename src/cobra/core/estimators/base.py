"""
Estimator module for expert pool learning in the COBRA pipeline.

This module defines the estimator layer, which forms the expert pool
used to generate candidate predictions before distance computation,
kernel weighting, and aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Estimators are base predictive models (experts) trained on the input data.
Their predictions are later:

- compared using distance metrics
- filtered by kernel functions
- combined through aggregation strategies

This design enables ensemble-like behavior where multiple estimators
contribute to a final consensus prediction.

Typical estimator types include:

- linear regression models
- decision trees
- k-nearest neighbors
- neural networks
- custom task-specific regressors/classifiers

By isolating estimators into a factory system, the framework supports:

- modular expert pools
- dynamic model selection
- experiment reproducibility
- scalable ensemble construction

Examples
--------
>>> @EstimatorFactory.register("linear")
... class LinearEstimator(BaseEstimator):
...     def fit(self, x, y):
...         return self
...
...     def predict(self, x):
...         return x @ self.coef_

>>> model = EstimatorFactory.create("linear")
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseEstimator(ABC):
    """
    Abstract base class for all estimators in the expert pool.

    Estimators are supervised learning models that produce predictions
    used as candidate signals for COBRA aggregation.

    Pipeline role
    -------------
    Each estimator contributes:

    - fitted model parameters (via ``fit``)
    - prediction outputs (via ``predict``)

    These predictions are later compared and combined by the COBRA
    framework.

    Notes
    -----
    Subclasses must implement both ``fit`` and ``predict``.

    Estimators are typically lightweight wrappers around ML models
    (e.g., scikit-learn compatible estimators).

    Examples
    --------
    >>> class DummyEstimator(BaseEstimator):
    ...     def fit(self, x, y):
    ...         return self
    ...
    ...     def predict(self, x):
    ...         return np.zeros(len(x))
    """

    @abstractmethod
    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> "BaseEstimator":
        """
        Fit estimator to training data.

        Parameters
        ----------
        x : ArrayLike
            Training features.

        y : ArrayLike
            Target values.

        Returns
        -------
        BaseEstimator
            Fitted estimator instance (``self``). Implementations should
            return ``self`` to remain compatible with scikit-learn style APIs.

        Notes
        -----
        - ``x`` is typically an array of shape ``(n_samples, n_features)``.
        - ``y`` is typically a 1-D array of shape ``(n_samples,)`` or a 2-D
          array for multi-output estimators.
        Examples
        --------
        >>> model.fit(X_train, y_train)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: ArrayLike):
        """
        Generate predictions for input samples.

        Parameters
        ----------
        x : ArrayLike
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values. Expected shape is ``(n_samples,)`` for
            single-output regressors/classifiers or ``(n_samples, n_outputs)``
            for multi-output estimators.

        Examples
        --------
        >>> preds = model.predict(X_test)
        """
        raise NotImplementedError


class EstimatorFactory(BaseFactory):
    """
    Factory for ``BaseEstimator`` implementations.

    This registry-based factory enables dynamic selection of estimator
    implementations for expert pool construction.

    It is commonly used in:

    - ensemble learning pipelines
    - COBRA-style aggregation models
    - automated machine learning systems
    - YAML-configured experiments

    Examples
    --------
    >>> estimator = EstimatorFactory.create("linear")

    >>> estimator.fit(X_train, y_train)

    >>> preds = estimator.predict(X_test)
    """