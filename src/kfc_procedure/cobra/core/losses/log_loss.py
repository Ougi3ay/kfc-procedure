"""
Log Loss (Cross-Entropy Loss).

Used for probabilistic classification models.

Formula:
    L = -mean(y log(p) + (1-y) log(1-p))
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("log_loss", "cross_entropy")
class LogLoss(BaseLoss):
    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.clip(np.asarray(y_pred), 1e-12, 1 - 1e-12)

        return float(-np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        ))
