"""
Space normalization package for COBRA consensus projection.

This package defines the normalization layer used to project inputs
and model outputs into a shared consensus space before distance
computation and kernel weighting.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Space normalization ensures that heterogeneous representations
produced by multiple estimators become comparable in a unified
metric space.

It plays a critical role in COBRA-style pipelines by:

- aligning feature and prediction spaces
- stabilizing distance computations
- improving kernel weighting consistency
- enabling fair comparison across estimators

Without normalization, differences in scale between models can lead
to biased or unstable consensus behavior.

Design philosophy
-----------------
This package is designed to be:

- modular (swap normalization strategies easily)
- extensible (add custom projection logic)
- consistent (shared transformation interface)
- factory-driven (string-based configuration)
- compatible with ensemble pipelines

Available components
--------------------

Base interface
^^^^^^^^^^^^^^

- ``BaseSpaceNormalizer``
    Abstract interface for space projection strategies.

Factory system
^^^^^^^^^^^^^^

- ``SpaceNormalizerFactory``
    Registry-based factory for normalization methods.

Built-in normalizers
^^^^^^^^^^^^^^^^^^^^

- ``IdentitySpaceNormalizer``
    No transformation applied (identity mapping).

- ``GradientCOBRASpaceNormalizer``
    Normalizes model outputs using a shared constant.

- ``MixCOBRASpaceNormalizer``
    Independently normalizes input features and model outputs.

Examples
--------
>>> from cobra.core.spaces import SpaceNormalizerFactory

>>> normalizer = SpaceNormalizerFactory.create("mixcobra")

>>> Xn, Yn = normalizer.transform(X, model_outputs)

Exports
-------
All space normalization components are exposed for convenient use in
COBRA and MIXCOBRA pipeline configurations.
"""

from .base import (
    BaseSpaceNormalizer,
    SpaceNormalizerFactory,
)

from .builtin import (
    IdentitySpaceNormalizer,
    GradientCOBRASpaceNormalizer,
    MixCOBRASpaceNormalizer,
)

__all__ = [
    "BaseSpaceNormalizer",
    "SpaceNormalizerFactory",
    "IdentitySpaceNormalizer",
    "GradientCOBRASpaceNormalizer",
    "MixCOBRASpaceNormalizer",
]
