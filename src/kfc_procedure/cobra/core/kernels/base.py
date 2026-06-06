"""
Kernel module for COBRA framework.

This module defines the abstract interface for kernel functions and
provides a factory system for dynamic kernel registration.

In the COBRA pipeline, kernels are not standalone transformation
operators. Instead, they act as evaluation functions applied to
parameterized distance representations produced by Kernel Adapters.

This design separates concerns into three components:

1. Distance Layer
   - Computes raw pairwise distances between samples.

2. Kernel Adapter Layer
   - Injects learnable or tunable parameters into distance space.
   - Performs transformations such as bandwidth, alpha, beta, etc.
   - Produces adapted distance representations.

3. Kernel Layer
   - Maps adapted distances into similarity or weight matrices.
   - Defines the final kernel function used for aggregation.

This separation allows:
- modular optimization
- flexible kernel design
- consistent parameter control across models
- integration with COBRA aggregation and meta-learning

Kernel outputs are directly used as:
- weight matrices for base estimators
- similarity scores in aggregation rules
- optimization objectives in learning procedures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from kfc_procedure.cobra.core.factory import BaseFactory

class BaseKernel(ABC):
    """
    Abstract base class for kernel functions.

    This class defines the interface for transforming distance
    matrices into similarity or weighting matrices.

    Attributes
    ----------
    requires_grad : bool
        Indicates whether kernel supports differentiable optimization.

    mode : str
        Kernel behavior type:
        - "continuous": smooth kernel (e.g., RBF)
        - "compact": bounded support kernels
        - "discrete": non-continuous mappings

    params : dict
        Kernel hyperparameters stored internally.

    Methods
    -------
    __call__(D)
        Apply kernel transformation to a distance matrix.

    set_params(**params)
        Update kernel parameters.

    get_params()
        Retrieve kernel parameters.

    is_continuous()
        Check if kernel is continuous.

    is_discrete()
        Check if kernel is discrete.
    """

    requires_grad: bool = True
    mode: str = "continuous"

    def __init__(self, **kwargs):
        """
        Initialize kernel with hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Kernel-specific parameters (e.g., gamma, sigma).
        """
        self.params: Dict[str, Any] = dict(kwargs)

        for k, v in self.params.items():
            setattr(self, k, v)

    def set_params(self, **params) -> "BaseKernel":
        """
        Update kernel hyperparameters.

        Parameters
        ----------
        **params : dict
            Key-value pairs of parameters to update.

        Returns
        -------
        BaseKernel
            Self instance for chaining.
        """
        for k, v in params.items():
            setattr(self, k, v)
            self.params[k] = v
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Retrieve kernel parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility.

        Returns
        -------
        dict
            Copy of kernel parameters.
        """
        return dict(self.params)

    @abstractmethod
    def __call__(self, D: np.ndarray) -> np.ndarray:
        """
        Transform distance matrix into similarity/kernel matrix.

        Parameters
        ----------
        D : np.ndarray
            Distance matrix of shape (n_samples, n_samples).

        Returns
        -------
        np.ndarray
            Kernel (similarity) matrix.
        """
        raise NotImplementedError

    def is_continuous(self) -> bool:
        """
        Check whether kernel is continuous.

        Returns
        -------
        bool
            True if kernel is continuous.
        """
        return self.mode == "continuous"

    def is_discrete(self) -> bool:
        """
        Check whether kernel is discrete.

        Returns
        -------
        bool
            True if kernel is discrete.
        """
        return self.mode == "discrete"

class KernelFactory(BaseFactory):
    """
    Factory for kernel functions.

    Enables dynamic registration and creation of kernel functions
    used in COBRA similarity learning pipeline.
    """
    pass