"""
Huber Loss.

A robust loss combining MSE and MAE behavior.

- Quadratic for small errors
- Linear for large errors

Parameter:
    delta controls transition point
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("huber")
class HuberLoss(BaseLoss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        error = y_true - y_pred
        abs_error = np.abs(error)

        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        return float(np.mean(np.where(abs_error <= self.delta, quadratic, linear)))
