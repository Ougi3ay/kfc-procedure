"""
Gradient optimization package for COBRA continuous tuning.

This package provides gradient-based optimization methods used to
tune continuous hyperparameters across the COBRA pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Gradient optimizers are used to minimize black-box or differentiable
objective functions involving continuous parameters such as:

- kernel bandwidth and decay rates
- distance scaling factors
- adapter coefficients (alpha, beta)
- smooth weighting parameters

These optimizers perform iterative updates to reduce loss and improve
overall ensemble performance.

Design philosophy
-----------------
This package is designed to be:

- extensible (add new gradient methods easily)
- modular (plug-and-play optimizers)
- consistent (shared base interface)
- experiment-friendly (factory-based selection)

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseGradientOptimizer``
    Abstract base class defining gradient update behavior.

Factory system
^^^^^^^^^^^^^^

- ``GradientOptimizerFactory``
    Registry-based factory for gradient optimizers.

Built-in optimizers
^^^^^^^^^^^^^^^^^^^^

- ``GradientDescentOptimizer``
    Finite-difference gradient descent implementation.

Examples
--------
>>> from cobra.core.optimizers.gradient import GradientOptimizerFactory

>>> optimizer = GradientOptimizerFactory.create("gradient_descent")

>>> best_params, history = optimizer(objective_fn, init_params)

Exports
-------
All gradient optimization components are exposed for convenient
integration into COBRA training pipelines.
"""

from .base import (
    BaseGradientOptimizer,
    GradientOptimizerFactory,
)

from .gd import GradientDescentOptimizer

__all__ = [
    "BaseGradientOptimizer",
    "GradientOptimizerFactory",
    "GradientDescentOptimizer",
]