"""
Normalization module for COBRA framework.

This package provides a unified API for data normalization strategies
used throughout the COBRA pipeline.

Available components
--------------------
- BaseNormalizer
- NormalizerFactory
- StandardNormalizer
- MinMaxNormalizer

Typical usage
-------------
>>> from cobra.core.normalizers import NormalizerFactory
>>> norm = NormalizerFactory.create("standard")
>>> Xn = norm.fit_transform(X)
"""

from .base import BaseNormalizer, NormalizerFactory
from .standard import StandardNormalizer
from .minmax import MinMaxNormalizer

__all__ = [
    "BaseNormalizer",
    "NormalizerFactory",
    "StandardNormalizer",
    "MinMaxNormalizer",
]
