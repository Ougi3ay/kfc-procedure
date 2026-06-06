"""
Cross-validation module for COBRA framework.

This package provides multiple cross-validation strategies:

- KFoldCV: standard k-fold for i.i.d. data
- StratifiedKFoldCV: preserves class distribution
- TimeSeriesCV: chronological validation for temporal data

All strategies are factory-registered for dynamic usage.
"""

from __future__ import annotations

from .base import BaseCrossValidator, CVFactory
from .kfold import KFoldCV
from .stratified_kfold import StratifiedKFoldCV
from .time_series import TimeSeriesCV


__all__ = [
    "BaseCrossValidator",
    "CVFactory",
    "KFoldCV",
    "StratifiedKFoldCV",
    "TimeSeriesCV",
]