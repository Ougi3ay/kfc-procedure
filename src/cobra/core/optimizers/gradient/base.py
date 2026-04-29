"""
Gradient optimizer module for continuous parameter tuning in COBRA.

This module defines the gradient-based optimization layer used to
tune continuous hyperparameters in the COBRA pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
Gradient optimizers are used when the objective function depends on
continuous parameters such as:

- kernel bandwidth (gamma, sigma)
- distance scaling factors
- adapter weights (alpha, beta)
- smoothness or regularization coefficients

Unlike discrete search methods, gradient-based optimizers iteratively
update parameters using local directional information.

Design goals
------------
- extend base optimizer interface
- support iterative parameter updates
- provide step-wise optimization abstraction
- enable plug-in gradient methods (GD, Adam, etc.)
- remain compatible with COBRA objective functions

Examples
--------
>>> @GradientOptimizerFactory.register("gd")
... class GradientDescent(BaseGradientOptimizer):
...     def step(self, objective, params):
...         grad = compute_grad(objective, params)
...         return params - self.lr * grad
...
...     def __call__(self, objective, params):
...         for _ in range(self.max_iter):
...             params = self.step(objective, params)
...         return params
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import numpy as np

from cobra.core.factory import BaseFactory
from cobra.core.optimizers.base import BaseOptimizer


class BaseGradientOptimizer(BaseOptimizer):
    """
    Abstract base class for gradient-based optimizers.

    This class extends ``BaseOptimizer`` by introducing a step-wise
    update rule for continuous parameter optimization.

    Pipeline role
    -------------
    Gradient optimizers refine:

    - kernel parameters
    - distance scaling coefficients
    - adapter hyperparameters
    - smooth weighting functions

    Notes
    -----
    Subclasses must implement:

    - ``step()``: single update rule
    - ``__call__()``: full optimization loop

    Examples
    --------
    >>> class SGDOptimizer(BaseGradientOptimizer):
    ...     def step(self, objective, params):
    ...         return params - 0.01
    """

    @abstractmethod
    def step(
        self,
        objective: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Perform a single optimization step.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize.

        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        np.ndarray
            Updated parameter vector after one step.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Examples
        --------
        >>> new_params = optimizer.step(objective, params)
        """
        raise NotImplementedError


class GradientOptimizerFactory(BaseFactory):
    """
    Factory for gradient-based optimizer implementations.

    This registry-based factory enables dynamic selection of gradient
    optimization algorithms for continuous parameter tuning.

    It is commonly used in:

    - hyperparameter optimization of COBRA components
    - continuous kernel tuning
    - adapter weight calibration
    - experimental optimization pipelines

    Examples
    --------
    >>> optimizer = GradientOptimizerFactory.create("gd")

    >>> best_params = optimizer(objective_fn, init_params)
    """
    pass
