"""
Kernel module for similarity weighting in the COBRA pipeline.

This module defines the kernel layer, which transforms adapted distance
matrices into similarity weights used for neighbor selection and
final aggregation.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
The kernel stage converts transformed distances into similarity scores
or weights that determine how much each neighbor contributes to the
final prediction.

Unlike distance metrics (which measure dissimilarity), kernels:

- convert distances into similarity space
- emphasize local neighborhoods
- control smoothness of the prediction function
- enable non-linear weighting of experts

Typical kernel outputs:

- similarity matrices
- weight distributions
- neighborhood importance scores

Design goals
------------
- modular kernel implementations
- interchangeable kernel functions
- compatibility with optimization routines
- factory-based instantiation for experiments

Examples
--------
>>> @KernelFactory.register("gaussian")
... class GaussianKernel(BaseKernel):
...     def __call__(self, distances):
...         return np.exp(-distances ** 2)

>>> kernel = KernelFactory.create("gaussian")
>>> weights = kernel(distance_matrix)
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Literal

import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    Base class for all COBRA kernel functions.

    Kernels convert distance matrices into similarity weights used
    for aggregation in the COBRA / GradientCOBRA pipeline.

    Two orthogonal concepts are supported:

    1. mode:
        - continuous: smooth weighting (RBF, Laplace)
        - discrete: hard selection (Indicator, COBRA voting)

    2. requires_grad:
        - whether kernel supports gradient-based optimization
    """

    requires_grad: bool = True
    mode: str = "continuous"  # continuous | compact | discrete

    def __init__(self, **kwargs):
        self.params: Dict[str, Any] = dict(kwargs)

        for k, v in self.params.items():
            setattr(self, k, v)

    def set_params(self, **params) -> "BaseKernel":
        """
        Update kernel hyperparameters.
        """
        for k, v in params.items():
            setattr(self, k, v)
            self.params[k] = v
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return kernel parameters (sklearn-compatible).
        """
        return dict(self.params)

    @abstractmethod
    def __call__(self, D: np.ndarray) -> np.ndarray:
        """
        Transform distance matrix into similarity weights.

        Parameters
        ----------
        D : np.ndarray
            Distance matrix of shape (n_samples, n_samples)
            or (n_queries, n_references)

        Returns
        -------
        np.ndarray
            Kernel weight matrix (same shape as D input).
        """
        raise NotImplementedError

    def is_continuous(self) -> bool:
        return self.mode == "continuous"

    def is_discrete(self) -> bool:
        return self.mode == "discrete"


class KernelFactory(BaseFactory):
    """
    Factory for kernel implementations.

    This registry-based factory enables dynamic creation of kernel
    functions using string identifiers.

    It is used in:

    - COBRA-style ensemble pipelines
    - hyperparameter optimization
    - YAML-based configuration systems
    - kernel benchmarking experiments

    Examples
    --------
    >>> kernel = KernelFactory.create("gaussian")

    >>> weights = kernel(distance_matrix)
    """
    pass