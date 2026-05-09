"""
Optimizer module for COBRA hyperparameter search and model tuning.

This module defines a unified optimization interface and multiple
optimization strategies used across the COBRA pipeline.

Design principles
-----------------
- unified API across optimizers
- vector-based parameter representation (np.ndarray)
- structured result output
- extensibility via factory pattern
- compatibility with both discrete and continuous optimization

All optimizers return a dictionary with:
    {
        "x": np.ndarray        # best parameter vector
        "score": float         # best objective value
        "history": list        # optimization trace
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict

import numpy as np
from itertools import product

from cobra.core.factory import BaseFactory

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

class BaseOptimizer(ABC):
    """
    Abstract base class for optimization strategies.

    All optimizers must implement the `optimize` method.

    Parameters
    ----------
    show_process : bool, default=True
        Whether to display progress bar.

    **kwargs : dict
        Additional optimizer-specific parameters.
    """

    def __init__(self, show_process: bool = True, **kwargs):
        self.show_process = show_process
        self.config = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param: np.ndarray | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization process.

        Parameters
        ----------
        objective : callable
            Function mapping parameters → scalar loss

        init_param : np.ndarray, optional
            Initial parameter vector (used in gradient-based methods)

        Returns
        -------
        dict
            Optimization result:
            {
                "x": best parameters,
                "score": best loss,
                "history": list of iteration records
            }
        """
        raise NotImplementedError

    # Backward compatibility
    def __call__(self, objective, init_param=None):
        return self.optimize(objective, init_param)

class OptimizerFactory(BaseFactory):
    """Factory for registering and creating optimizers."""
    pass
