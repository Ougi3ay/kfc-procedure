"""
Mean Absolute Error Loss.

This loss measures the average absolute difference between
true and predicted values.

It is more robust to outliers than MSE.

Formula:
    L = mean(|y_true - y_pred|)
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("mae", "l1")
class MAELoss(BaseLoss):
    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))
