"""
Optimizers module for COBRA framework.

This module provides all optimization strategies used in COBRA,
including:

- Gradient-based optimizers (GD, Momentum, Adam)
- Search-based optimizers (Grid Search, Random Search)

These optimizers are used to tune:
- kernel parameters
- adapter weights
- loss functions
- model hyperparameters

All optimizers share a unified interface via BaseOptimizer
and are accessible through OptimizerFactory.
"""

from __future__ import annotations

from .base import BaseOptimizer, OptimizerFactory

from .gradient import (
    BaseGradientOptimizer,
    GradientDescentOptimizer,
    MomentumOptimizer,
    AdamOptimizer,
)

from .search import (
    BaseSearchOptimizer,
    GridSearchOptimizer,
)

__all__ = [
    # base
    "BaseOptimizer",
    "OptimizerFactory",

    # gradient-based
    "BaseGradientOptimizer",
    "GradientDescentOptimizer",
    "MomentumOptimizer",
    "AdamOptimizer",

    # search-based
    "BaseSearchOptimizer",
    "GridSearchOptimizer",
]