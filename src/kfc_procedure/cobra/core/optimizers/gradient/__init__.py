"""
Gradient-based optimizers module for COBRA framework.

This package provides implementations of gradient-based optimization
algorithms used in COBRA for:

- kernel parameter learning
- adapter weight optimization
- loss minimization
- general continuous parameter tuning

Supported optimizers:
---------------------
- GradientDescentOptimizer (GD): baseline optimizer
- MomentumOptimizer: accelerated gradient descent with velocity
- AdamOptimizer: adaptive moment estimation (recommended default)

All optimizers follow a unified interface defined in:
BaseGradientOptimizer
"""

from __future__ import annotations

from .base import BaseGradientOptimizer
from .gd import GradientDescentOptimizer
from .adam import AdamOptimizer
from .momentum import MomentumOptimizer


__all__ = [
    "BaseGradientOptimizer",
    "GradientDescentOptimizer",
    "AdamOptimizer",
    "MomentumOptimizer",
]
