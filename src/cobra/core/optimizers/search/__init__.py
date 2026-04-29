"""
Search optimization package for discrete hyperparameter exploration.

This package defines the search-based optimization layer used in the
COBRA pipeline for exploring discrete, categorical, or structured
parameter spaces.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Search optimizers are used when model parameters cannot be optimized
using gradients. This includes:

- categorical choices (kernel type, estimator type)
- discrete hyperparameters (k in kNN, tree depth)
- structured configurations (model pipelines)
- black-box parameter spaces

Instead of gradient updates, these methods rely on:

- grid enumeration
- random sampling
- heuristic exploration strategies

Design philosophy
-----------------
This package is designed to be:

- modular (plug-in search strategies)
- extensible (custom sampling logic)
- reproducible (deterministic search options)
- factory-driven (string-based configuration)
- compatible with COBRA objective evaluation

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseSearchOptimizer``
    Abstract interface for search-based optimization.

Factory system
^^^^^^^^^^^^^^

- ``SearchOptimizerFactory``
    Registry-based factory for search optimizers.

Built-in optimizers
^^^^^^^^^^^^^^^^^^^

- ``GridSearchOptimizer``
    Exhaustive search over a predefined parameter grid.

Examples
--------
>>> from cobra.core.optimizers.search import SearchOptimizerFactory

>>> optimizer = SearchOptimizerFactory.create("grid_search")

>>> config = optimizer.sample()

Exports
-------
All search-based optimization components are exposed for convenient
use in configuration-driven COBRA pipelines.
"""

from .base import (
    BaseSearchOptimizer,
    SearchOptimizerFactory,
)

from .grid import GridSearchOptimizer

__all__ = [
    "BaseSearchOptimizer",
    "SearchOptimizerFactory",
    "GridSearchOptimizer",
]
