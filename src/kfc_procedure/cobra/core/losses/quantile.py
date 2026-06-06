"""
Quantile Loss.

Used in quantile regression.

Parameter:
    tau ∈ (0, 1) controls quantile level.

Formula:
    L = max(tau*(y - y_pred), (tau-1)*(y - y_pred))
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("quantile")
class QuantileLoss(BaseLoss):
    def __init__(self, tau: float = 0.5):
        self.tau = tau

    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        error = y_true - y_pred

        return float(np.mean(
            np.maximum(self.tau * error, (self.tau - 1) * error)
        ))
