"""
Search-based optimizers module for COBRA framework.

This module provides derivative-free optimization strategies that
evaluate a predefined or generated set of candidate solutions.

Search optimizers are used in COBRA for:
- hyperparameter tuning
- kernel selection
- adapter configuration search
- model comparison without gradients

Unlike gradient-based optimizers, search methods:
- do not require derivatives
- operate on discrete candidate sets
- are highly parallelizable
"""

from __future__ import annotations

from .base import BaseSearchOptimizer
from .search import GridSearchOptimizer


__all__ = [
    "BaseSearchOptimizer",
    "GridSearchOptimizer",
]
