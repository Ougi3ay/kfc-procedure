"""
Optimizer module for COBRA hyperparameter search and model tuning.

This module defines the optimization layer used to tune components
across the COBRA pipeline, including:

- estimators
- distance metrics
- kernel adapters
- kernel functions
- loss functions

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Optimizers are responsible for minimizing a given objective function
that evaluates model performance.

In COBRA-style systems, the objective typically depends on:

- kernel parameters
- distance scaling factors
- estimator configurations
- aggregation behavior

The optimizer searches for parameter settings that minimize loss.

Design goals
------------
- support generic objective functions
- enable plug-and-play optimization strategies
- allow iterative or gradient-free search methods
- provide consistent callable interface
- support progress tracking (optional tqdm integration)

Examples
--------
>>> class GridSearchOptimizer(BaseOptimizer):
...     def __call__(self, objective, grid):
...         best_score = float("inf")
...         best_params = None
...         for params in grid:
...             score = objective(params)
...             if score < best_score:
...                 best_score = score
...                 best_params = params
...         return best_params
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = lambda x, **kwargs: range(x)


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization strategies.

    Optimizers search over a parameter space to minimize an objective
    function used in the COBRA pipeline.

    Pipeline role
    -------------
    Optimizers tune:

    - kernel hyperparameters
    - distance metrics
    - adapter weights
    - estimator configurations
    - loss-related parameters

    Attributes
    ----------
    dynamic attributes : Any
        Optimizer-specific hyperparameters passed via constructor.

    Notes
    -----
    Subclasses must implement the ``__call__`` method, which defines
    the optimization procedure.

    Examples
    --------
    >>> optimizer = MyOptimizer(max_iter=100)
    >>> best = optimizer(objective_fn)
    """

    def __init__(self, show_process=True, **kwargs):
        """
        Initialize optimizer with hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Optimizer configuration parameters.
        """
        self.show_process = show_process
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """
        Return string representation of optimizer.

        Returns
        -------
        str
            Human-readable optimizer configuration.
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        return f"{self.__class__.__name__}({attrs})"

    @abstractmethod
    def __call__(
        self,
        objective: Callable[[np.ndarray], float],
        *args,
        **kwargs,
    ):
        """
        Execute optimization procedure.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Function that evaluates a parameter configuration
            and returns a scalar loss value.

        *args, **kwargs
            Additional optimizer-specific arguments.

        Returns
        -------
        Any
            Best found solution (implementation dependent).

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> best = optimizer(objective_fn)
        """
        raise NotImplementedError
