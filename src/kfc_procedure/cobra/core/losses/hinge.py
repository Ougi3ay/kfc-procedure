"""
Hinge Loss.

Commonly used in SVM-style classification.

Formula:
    L = mean(max(0, 1 - y_true * y_pred))

Assumes:
    y_true in {-1, +1}
"""

from __future__ import annotations
import numpy as np

from .base import BaseLoss, LossFactory


@LossFactory.register("hinge")
class HingeLoss(BaseLoss):
    def __call__(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return float(np.mean(np.maximum(0.0, 1.0 - y_true * y_pred)))
