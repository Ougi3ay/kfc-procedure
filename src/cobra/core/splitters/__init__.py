"""
Data splitting package for COBRA training and calibration workflows.

This package defines the splitting layer responsible for partitioning
datasets into subsets used throughout the COBRA pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Data splitting is a foundational component in COBRA-style systems,
ensuring that learning, calibration, and evaluation operate on
properly separated data partitions.

It is responsible for:

- separating training and calibration sets
- enabling cross-validation workflows
- supporting ensemble evaluation strategies
- preserving index alignment across pipeline stages
- ensuring reproducible experimental setups

Design philosophy
-----------------
This package is designed to be:

- modular (multiple splitting strategies)
- extensible (custom splitter implementations)
- reproducible (seed-controlled randomness)
- index-based (avoids data duplication issues)
- factory-driven (configuration-based selection)

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseDataSplitter``
    Abstract interface for dataset splitting strategies.

Factory system
^^^^^^^^^^^^^^

- ``SplitterFactory``
    Registry-based factory for splitter implementations.

Built-in splitters
^^^^^^^^^^^^^^^^^^

- ``RandomHoldoutSplitter``
    Random train/calibration partitioning.

- ``KFoldSplitter``
    K-fold cross-validation splitting.

Examples
--------
>>> from cobra.core.splitters import SplitterFactory

>>> splitter = SplitterFactory.create("holdout")
>>> train_idx, cal_idx = splitter.split(X, y)

>>> splitter = SplitterFactory.create("kfold")
>>> folds = splitter.split(X, y)

Exports
-------
All splitter components are exposed for use in COBRA pipeline
configuration and experimentation frameworks.
"""

from .base import BaseDataSplitter
from .base import SplitterFactory
from .builtin import KFoldSplitter, RandomHoldoutSplitter

__all__ = [
    "BaseDataSplitter",
    "SplitterFactory",
    "RandomHoldoutSplitter",
    "KFoldSplitter",
]