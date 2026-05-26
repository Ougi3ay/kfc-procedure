"""
kfc_procedure.core.ml
---------------------

Unified ML layer for F-step local models.

This module:
1. Defines base interfaces
2. Registers built-in models
3. Auto-registers sklearn estimators
"""

from __future__ import annotations

from .base import BaseLocalModel, LocalModelFactory
from .builtin import DummyRegressor, register_all_sklearn_models

# Trigger sklearn registration on import
register_all_sklearn_models()


__all__ = [
    "BaseLocalModel",
    "LocalModelFactory",
    "DummyRegressor",
]