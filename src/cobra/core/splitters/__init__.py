"""
Data splitting strategies for COBRA-based learning frameworks.

This package provides abstractions and concrete implementations for
partitioning datasets into training and evaluation subsets.

Available splitters include:

- RandomHoldoutSplitter
    Random train/evaluation partitioning using holdout sampling.

- OverlapSplitter
    Partitioning strategy supporting controlled overlap between
    training and evaluation subsets.

The package also exposes:

- BaseDataSplitter
    Abstract interface implemented by all splitting strategies.

- SplitterFactory
    Registry-based factory for dynamic splitter discovery and
    instantiation.

Examples
--------
Create a splitter directly:

>>> splitter = RandomHoldoutSplitter(
...     calibration_size=0.5,
...     random_state=42,
... )

Create a splitter through the factory:

>>> splitter = SplitterFactory.create(
...     "holdout",
...     calibration_size=0.5,
... )

List available splitters:

>>> SplitterFactory.available()
['holdout', 'random_holdout', 'split_overlap']
"""

from __future__ import annotations
from .base import (
    BaseDataSplitter,
    SplitterFactory,
)
from .holdout import RandomHoldoutSplitter
from .overlap import OverlapSplitter

__all__ = [
    "BaseDataSplitter",
    "SplitterFactory",
    "RandomHoldoutSplitter",
    "OverlapSplitter",
]