"""
COBRA utilities package.

This package provides shared helper functions used across the COBRA
pipeline for preprocessing and configuration resolution.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The utilities module acts as a support layer for the COBRA framework,
providing reusable functionality that simplifies pipeline construction
and execution.

It includes:

1. Preprocessing utilities
   - dataset splitting with overlap support
   - normalization constant computation for numerical stability

2. Resolver utilities
   - conversion of configuration strings into factory-based objects
   - dynamic instantiation of pipeline components

Design goals
------------
- reduce boilerplate in pipeline construction
- unify preprocessing and configuration logic
- support configuration-driven experimentation
- ensure consistency across all COBRA components
- maintain lightweight and dependency-minimal design

Submodules
----------

preprocessing
^^^^^^^^^^^^^
Functions for dataset manipulation and scaling:

- ``data_split_overlap``
    Creates overlapping dataset partitions for calibration workflows.

- ``compute_normalization_constant``
    Computes scaling constants for stabilizing feature magnitudes.

resolve
^^^^^^^
Factory resolver utilities that convert configuration inputs into
runtime objects:

- ``resolve_from_estimators``
- ``resolve_from_kernel``
- ``resolve_from_splitter``
- ``resolve_from_aggregator``
- ``resolve_from_loss``
- ``resolve_from_distance``

These functions enable declarative pipeline construction from
strings or configuration files.

Examples
--------
Preprocessing:

>>> X_k, y_k, X_l, y_l, idx_k, idx_l = data_split_overlap(X, y)

Resolvers:

>>> estimator = resolve_from_estimators("ridge", None, ["ridge"])
>>> kernel = resolve_from_kernel("rbf", {"gamma": 1.0})
"""

from .preprocessing import (
    data_split_overlap,
    compute_normalization_constant,
)

from .resolve import (
    resolve_from_estimators,
    resolve_from_kernel,
    resolve_from_splitter,
    resolve_from_aggregator,
    resolve_from_loss,
    resolve_from_distance,
)

__all__ = [
    "data_split_overlap",
    "compute_normalization_constant",
    "resolve_from_estimators",
    "resolve_from_kernel",
    "resolve_from_splitter",
    "resolve_from_aggregator",
    "resolve_from_loss",
    "resolve_from_distance",
]