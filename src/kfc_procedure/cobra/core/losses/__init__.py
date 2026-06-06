"""
Loss module for COBRA framework.

This package provides loss functions used for:
- regression evaluation
- classification evaluation
- robust optimization objectives
- COBRA aggregation tuning
"""

from __future__ import annotations
from .base import BaseLoss, LossFactory

from .mse import MSELoss
from .mae import MAELoss
from .huber import HuberLoss
from .log_loss import LogLoss
from .hinge import HingeLoss
from .quantile import QuantileLoss


__all__ = [
    "BaseLoss",
    "LossFactory",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "LogLoss",
    "HingeLoss",
    "QuantileLoss",
]