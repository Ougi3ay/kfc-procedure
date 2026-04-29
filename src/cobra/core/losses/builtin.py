"""
Built-in loss functions for COBRA optimization objectives.

This module provides standard loss functions used to evaluate and
optimize prediction performance in regression and classification
settings within the COBRA pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Loss functions define the optimization objective used to evaluate
how well the aggregated predictions match ground truth targets.

They are typically used for:

- hyperparameter tuning (kernel, adapter, distance, estimators)
- model selection across expert pools
- performance evaluation in validation loops
- training-time optimization objectives

Loss types
----------

1. MSELoss (Mean Squared Error)

    Measures squared deviation between predictions and targets.

2. MAELoss (Mean Absolute Error)

    Measures absolute deviation, more robust to outliers.

3. ZeroOneLoss

    Classification error measuring misclassification rate.

Design note
-----------
These losses are designed to be:

- simple and numerically stable
- compatible with optimization routines
- applicable in ensemble evaluation stages

Examples
--------
>>> loss_fn = LossFactory.create("mse")
>>> loss_fn(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseLoss, LossFactory


@LossFactory.register("mse", "mean_squared_error")
class MSELoss(BaseLoss):
    """
    Mean Squared Error (MSE) loss.

    Computes average squared difference between true and predicted values.

    Mathematical form
    -----------------
    L = mean((y_true - y_pred)^2)

    Properties
    ----------
    - penalizes large errors heavily
    - smooth and differentiable
    - commonly used in regression tasks

    Examples
    --------
    >>> loss = MSELoss()
    >>> loss(y_true, y_pred)
    """

    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> float:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)

        return float(np.mean((yt - yp) ** 2))


@LossFactory.register("mae", "mean_absolute_error")
class MAELoss(BaseLoss):
    """
    Mean Absolute Error (MAE) loss.

    Computes average absolute difference between true and predicted values.

    Mathematical form
    -----------------
    L = mean(|y_true - y_pred|)

    Properties
    ----------
    - robust to outliers
    - linear error penalty
    - widely used in regression evaluation

    Examples
    --------
    >>> loss = MAELoss()
    >>> loss(y_true, y_pred)
    """

    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> float:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)

        return float(np.mean(np.abs(yt - yp)))


@LossFactory.register("zero_one", "classification_error")
class ZeroOneLoss(BaseLoss):
    """
    Zero-One loss (classification error rate).

    Measures proportion of incorrect predictions.

    Mathematical form
    -----------------
    L = mean(y_true != y_pred)

    Properties
    ----------
    - strict correctness measure
    - commonly used in classification evaluation
    - non-differentiable

    Examples
    --------
    >>> loss = ZeroOneLoss()
    >>> loss(y_true, y_pred)
    """

    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> float:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)

        return float(np.mean(yt != yp))
