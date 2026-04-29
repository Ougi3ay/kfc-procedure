"""
Optimizer package for COBRA pipeline parameter tuning.

This package defines the optimization layer used across the COBRA
framework to tune model components such as:

- estimators
- distance metrics
- kernel adapters
- kernel functions
- loss objectives

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Optimizers are responsible for searching or refining parameters that
minimize a given objective function. This package supports both:

1. Continuous optimization (gradient-based)
2. Discrete/structured search (grid, sampling)

Together, they provide a unified interface for all COBRA tuning
strategies.

Design philosophy
-----------------
This package is designed to be:

- modular (swap optimization strategies easily)
- extensible (add custom optimizers)
- unified (shared BaseOptimizer interface)
- factory-driven (string-based configuration)
- compatible with black-box objectives

Optimization families
---------------------

Gradient-based optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Used for continuous parameters (e.g., gamma, alpha):

- ``BaseGradientOptimizer``
- ``GradientOptimizerFactory``
- ``GradientDescentOptimizer``

Search-based optimizers
^^^^^^^^^^^^^^^^^^^^^^^

Used for discrete or categorical parameters:

- ``BaseSearchOptimizer``
- ``SearchOptimizerFactory``
- ``GridSearchOptimizer``

Base interface
^^^^^^^^^^^^^^

- ``BaseOptimizer``
    Common abstraction for all optimizer types.

Examples
--------
Gradient optimization:

>>> optimizer = GradientOptimizerFactory.create("gradient_descent")
>>> params, history = optimizer(objective_fn, init_params)

Grid search:

>>> optimizer = SearchOptimizerFactory.create("grid_search")
>>> best_params, history = optimizer(objective_fn)

Exports
-------
All optimizer components are exposed for easy integration into COBRA
configuration pipelines.
"""

from .base import BaseOptimizer

from .gradient.base import (
    BaseGradientOptimizer,
    GradientOptimizerFactory,
)
from .gradient.gd import GradientDescentOptimizer

from .search.base import (
    BaseSearchOptimizer,
    SearchOptimizerFactory,
)
from .search.grid import GridSearchOptimizer

__all__ = [
    "BaseOptimizer",

    "BaseGradientOptimizer",
    "GradientOptimizerFactory",
    "GradientDescentOptimizer",

    "BaseSearchOptimizer",
    "SearchOptimizerFactory",
    "GridSearchOptimizer",
]