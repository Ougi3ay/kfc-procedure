"""
Optimizer module for COBRA framework.

This module defines the abstract optimization interface used to
tune parameters in the COBRA pipeline.

Optimizers are responsible for solving:

    argmin_theta  objective(theta)

where theta can represent:
- kernel parameters
- adapter weights
- distance scaling factors
- ensemble aggregation coefficients

This is a core component of:
- kernel learning
- model selection
- aggregation optimization
- cross-validation-based tuning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict

import numpy as np

from kfc_procedure.cobra.core.factory import BaseFactory

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class BaseOptimizer(ABC):
    """
    Abstract optimizer interface for COBRA.

    All optimization strategies must implement this interface.

    Parameters
    ----------
    show_process : bool
        Whether to display progress bar during optimization.

    Attributes
    ----------
    config : dict
        Hyperparameters of the optimizer.
    """

    def __init__(self, show_process: bool = True, **kwargs):
        self.show_process = show_process
        self.config: Dict[str, Any] = dict(kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param: np.ndarray | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization procedure.

        Parameters
        ----------
        objective : Callable
            Function to minimize: f(theta) -> loss

        init_param : np.ndarray, optional
            Initial parameter vector.

        Returns
        -------
        Dict[str, Any]
            Optimization result containing:
            - best_params
            - best_score
            - history (optional)
        """
        raise NotImplementedError

    def __call__(self, objective, init_param=None):
        return self.optimize(objective, init_param)


class OptimizerFactory(BaseFactory):
    """
    Factory for registering and instantiating optimizers.

    Enables dynamic selection of optimization strategies in COBRA:

    Example
    -------
    >>> opt = OptimizerFactory.create("gradient_descent", lr=0.01)
    >>> result = opt.optimize(objective)
    """
    pass
