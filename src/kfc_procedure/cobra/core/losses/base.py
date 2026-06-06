"""
Loss module for COBRA framework.

This module defines the abstract interface for loss functions and
provides a factory system for dynamic loss registration.

In the COBRA pipeline, loss functions are used to:
- evaluate prediction quality
- guide optimization of kernel/adapters
- compare aggregated outputs with ground truth

Loss functions are a key component in the optimization loop,
linking model predictions to learning signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


# =========================================================
# Base Loss
# =========================================================
class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    This class defines the interface for evaluating the difference
    between true labels and predicted outputs.

    Loss functions are used in:
    - model evaluation
    - optimization objectives
    - cross-validation scoring

    Methods
    -------
    __call__(y_true, y_pred)
        Compute scalar loss value.
    """

    @abstractmethod
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> float:
        """
        Compute loss value between true and predicted outputs.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth labels.

        y_pred : ArrayLike
            Model predictions.

        Returns
        -------
        float
            Scalar loss value.
        """
        raise NotImplementedError


# =========================================================
# Loss Factory
# =========================================================
class LossFactory(BaseFactory):
    """
    Factory for loss functions.

    Enables dynamic registration and creation of loss functions
    used in COBRA optimization and evaluation pipeline.
    """
    pass
