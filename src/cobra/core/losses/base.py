"""
Loss module for optimization in the COBRA pipeline.

This module defines the loss layer, which quantifies the error between
predicted outputs and true targets during model optimization.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Loss functions provide a scalar objective that guides:

- hyperparameter optimization
- model selection across expert pools
- kernel and adapter tuning
- evaluation of ensemble performance

In COBRA-style systems, the loss is typically computed after:

- kernel weighting
- aggregation of neighbor predictions

and is used to measure final prediction quality.

Design goals
------------
- simple and interchangeable loss definitions
- compatibility with optimization routines
- extensible for custom metrics
- factory-based selection for experiments

Examples
--------
>>> @LossFactory.register("mse")
... class MeanSquaredError(BaseLoss):
...     def __call__(self, y_true, y_pred):
...         return float(np.mean((y_true - y_pred) ** 2))

>>> loss_fn = LossFactory.create("mse")
>>> loss = loss_fn(y_true, y_pred)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    Loss functions compute a scalar error between true and predicted
    values, used for optimization and model evaluation.

    Pipeline role
    -------------
    Loss functions guide:

    - parameter tuning
    - kernel selection
    - estimator weighting
    - ensemble optimization

    Notes
    -----
    Subclasses must implement ``__call__``.

    Loss functions are typically used in optimization loops or
    validation stages.

    Examples
    --------
    >>> class MAELoss(BaseLoss):
    ...     def __call__(self, y_true, y_pred):
    ...         return np.mean(np.abs(y_true - y_pred))
    """

    @abstractmethod
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> float:
        """
        Compute scalar loss value.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target values.

        y_pred : ArrayLike
            Predicted values.

        Returns
        -------
        float
            Scalar loss value.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> loss(y_true, y_pred)
        0.25
        """
        raise NotImplementedError


class LossFactory(BaseFactory):
    """
    Factory for loss function implementations.

    This registry-based factory enables dynamic selection of loss
    functions for optimization and evaluation in COBRA pipelines.

    It is commonly used in:

    - hyperparameter tuning loops
    - automated machine learning pipelines
    - model benchmarking frameworks
    - configuration-driven training systems

    Examples
    --------
    >>> loss_fn = LossFactory.create("mse")

    >>> value = loss_fn(y_true, y_pred)
    """
