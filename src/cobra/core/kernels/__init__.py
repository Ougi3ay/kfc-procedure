"""
Kernel module for COBRA framework.

This package provides a collection of kernel functions used to
transform adapted distance representations into similarity
or weight matrices.

Kernels are a core component of the COBRA pipeline and are used in:
- similarity learning
- aggregation of base estimators
- kernel-based optimization procedures

The module is fully factory-driven, enabling dynamic registration
and instantiation of kernel functions.
"""

from __future__ import annotations

from .base import BaseKernel, KernelFactory

from .reverse_cosh import ReverseCoshKernel
from .exponential import ExponentialKernel
from .radial import RadialKernel
from .cauchy import CauchyKernel
from .epanechnikov import EpanechnikovKernel
from .biweight import BiweightKernel
from .triweight import TriweightKernel
from .triangular import TriangularKernel
from .naive import NaiveKernel
from .cobra import COBRAKernel


__all__ = [
    "BaseKernel",
    "KernelFactory",

    "ReverseCoshKernel",
    "ExponentialKernel",
    "RadialKernel",
    "CauchyKernel",

    "EpanechnikovKernel",
    "BiweightKernel",
    "TriweightKernel",
    "TriangularKernel",

    "NaiveKernel",
    "COBRAKernel",
]
