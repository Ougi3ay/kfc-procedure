"""
Mean Squared Error Loss.

This loss measures the average squared difference between
true and predicted values.

It is the most common regression loss and strongly penalizes
large errors.

Formula:
    L = mean((y_true - y_pred)^2)
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("mse", "l2", "squared_error")
class MSELoss(BaseLoss):
    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))
