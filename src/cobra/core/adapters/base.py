"""
Kernel Adapter module for COBRA framework.

This module defines a parametric transformation layer that operates
on one or more distance matrices before they are passed into kernel
construction or optimization components.

The Kernel Adapter acts as an interface between:
- Distance metrics (geometry space)
- Kernel functions (similarity mapping)
- Optimization procedures (parameter tuning)

It enables COBRA to support learnable or tunable transformations
over distance representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from cobra.core.factory import BaseFactory

class BaseKernelAdapter(ABC):
    """
    Abstract base class for kernel transformation adapters.

    This class defines a unified interface for transforming one or
    more distance matrices into adapted representations used by
    kernel functions.

    The adapter may contain learnable or tunable parameters that
    can be optimized by external optimization procedures.

    Attributes
    ----------
    params : dict
        Dictionary of all adapter parameters.

    Methods
    -------
    transform(*distances)
        Transform input distance matrices into adapted representation.

    get_params()
        Return current parameters.

    set_params(**params)
        Update parameters dynamically.

    parameter_vector()
        Return parameters as a numeric vector for optimization.
    """

    def __init__(self, **kwargs):
        """
        Initialize kernel adapter with parameters.

        Parameters
        ----------
        **kwargs : dict
            Initial parameter values (e.g., bandwidth, alpha, beta).
        """
        self.params: Dict[str, Any] = dict(kwargs)
        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        Update adapter parameters.

        Parameters
        ----------
        **params : dict
            Key-value pairs of parameters to update.

        Returns
        -------
        BaseKernelAdapter
            Self for chaining.
        """
        for k, v in params.items():
            setattr(self, k, v)

        self.params.update(params)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Retrieve adapter parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn-style compatibility.

        Returns
        -------
        dict
            Copy of internal parameter dictionary.
        """
        return dict(self.params)

    def parameter_vector(self) -> np.ndarray:
        """
        Convert parameters into numeric vector form.

        This is used by optimizers to update adapter parameters.

        Returns
        -------
        np.ndarray
            Vector representation of parameters.
        """
        return np.array(list(self.params.values()), dtype=float)

    @abstractmethod
    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Transform one or more distance matrices.

        Parameters
        ----------
        *distances : np.ndarray
            One or more distance matrices.

        Returns
        -------
        np.ndarray
            Transformed distance representation.
        """
        raise NotImplementedError
class KernelAdapterFactory(BaseFactory):
    """
    Factory for kernel adapters.

    Allows dynamic registration and creation of kernel transformation
    strategies used in COBRA pipeline.
    """
    pass
