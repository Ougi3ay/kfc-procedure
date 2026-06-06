"""
Kernel Adapter module for COBRA framework.

This package provides a set of transformation layers that operate
on distance matrices before kernel construction and optimization.

Kernel adapters serve as a bridge between:
- Distance metrics (geometry space)
- Kernel functions (similarity mapping)
- Optimization procedures (parameter tuning)

These adapters allow COBRA to support:
- single-parameter learnable parameters
- multi-parameter linear combination
"""

from __future__ import annotations

from .base import (
    BaseKernelAdapter,
    KernelAdapterFactory,
)
from .one_parameter import OneParameterKernelAdapter
from .two_parameter import TwoParameterKernelAdapter


__all__ = [
    "BaseKernelAdapter",
    "OneParameterKernelAdapter",
    "TwoParameterKernelAdapter",
    "KernelAdapterFactory",
]