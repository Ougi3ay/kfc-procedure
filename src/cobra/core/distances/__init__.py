"""
Distance package for COBRA neighbor selection.

This package provides the distance computation layer used in the
COBRA pipeline to measure similarity between samples before kernel
transformation and aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Distance metrics define how similarity or dissimilarity is measured
between samples. These values are later used for:

- neighbor selection
- kernel weighting
- consensus aggregation
- model optimization

By isolating distance logic into a dedicated module, the framework
supports:

- extensibility (custom metrics)
- plug-and-play configurations
- experiment reproducibility
- multi-metric pipelines

Available components
--------------------

Base classes
^^^^^^^^^^^^

- ``BaseDistance``
    Abstract interface for all distance metrics.

- ``DistanceFactory``
    Registry-based factory for dynamic metric selection.

Built-in implementations
^^^^^^^^^^^^^^^^^^^^^^^^

- ``EuclideanDistance`` (L2)
    Standard geometric distance.

- ``ManhattanDistance`` (L1)
    Sum of absolute coordinate differences.

- ``MinkowskiDistance`` (Lp)
    Generalized distance with configurable exponent.

- ``CosineDistance``
    Angular distance based on cosine similarity.

- ``HammingDistance``
    Fraction of differing coordinates.

Examples
--------
>>> from cobra.core.distances import DistanceFactory

>>> distance = DistanceFactory.create("euclidean")

>>> D = distance.matrix(X_train, X_test)

Exports
-------
All commonly used distance metrics are exposed for convenient import.
"""

from .base import (
    BaseDistance,
    DistanceFactory,
)

from .builtin import (
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
    CosineDistance,
    HammingDistance,
)

__all__ = [
    "BaseDistance",
    "DistanceFactory",
    "EuclideanDistance",
    "ManhattanDistance",
    "MinkowskiDistance",
    "CosineDistance",
    "HammingDistance",
]