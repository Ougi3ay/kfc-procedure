"""
Loss package for COBRA optimization and evaluation.

This package defines the loss layer of the COBRA pipeline, which is
responsible for scoring predictions during optimization and model
selection.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Loss functions quantify the discrepancy between predicted values and
ground truth targets. They are used to:

- guide hyperparameter optimization
- evaluate ensemble performance
- compare estimator pools
- select optimal kernel and adapter configurations

In COBRA-style systems, losses are typically computed after:

- kernel weighting
- aggregation of neighbor predictions

and are used as the final objective for model tuning.

Design philosophy
-----------------
This package is designed to be:

- modular (swap loss functions easily)
- extensible (add custom metrics)
- consistent (uniform callable interface)
- factory-driven (dynamic selection via strings)

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseLoss``
    Abstract interface defining the loss contract.

Factory system
^^^^^^^^^^^^^^

- ``LossFactory``
    Registry-based factory for dynamic loss selection.

Built-in losses
^^^^^^^^^^^^^^^

- ``MSELoss``
    Mean squared error (L2 regression loss).

- ``MAELoss``
    Mean absolute error (L1 regression loss).

- ``ZeroOneLoss``
    Classification error rate (0–1 loss).

Examples
--------
>>> from cobra.core.losses import LossFactory

>>> loss_fn = LossFactory.create("mse")

>>> value = loss_fn(y_true, y_pred)

Exports
-------
All commonly used loss functions are exposed for convenient import
and pipeline integration.
"""

from .base import (
    BaseLoss,
    LossFactory,
)

from .builtin import (
    MAELoss,
    MSELoss,
    ZeroOneLoss,
)

__all__ = [
    "BaseLoss",
    "LossFactory",
    "MSELoss",
    "MAELoss",
    "ZeroOneLoss",
]
