"""
Gradient Descent optimizer for COBRA framework.

This module implements the simplest gradient-based optimization
algorithm: standard Gradient Descent (GD).

Update rule:

    x_{t+1} = x_t - lr * grad(x_t)

This optimizer is used as:
- baseline optimization method
- sanity check for gradient correctness
- comparison point for advanced optimizers (Adam, Momentum, etc.)

It is commonly applied to:
- kernel parameter tuning
- adapter weight optimization
- COBRA aggregation loss minimization
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer


@OptimizerFactory.register(
    "gd",
    categories={"optimizer", "gradient"},
)
class GradientDescentOptimizer(BaseGradientOptimizer):
    """
    Standard Gradient Descent optimizer.

    This optimizer performs deterministic gradient updates without
    momentum, adaptive learning rates, or second-order information.

    Parameters inherited from BaseGradientOptimizer:
    ------------------------------------------------
    learning_rate : float
        Step size scaling factor.

    max_iter : int
        Maximum number of iterations.

    tol : float
        Stopping threshold for gradient norm.

    speed : str
        Learning rate schedule strategy.

    gradient_method : str
        Method used for gradient approximation.

    eps : float
        Numerical stability constant for finite differences.

    Notes
    -----
    - This is the simplest optimizer in COBRA.
    - It is primarily used as a baseline.
    - It assumes smooth objective functions.
    """

    def step(
        self,
        x: np.ndarray,
        lr: float,
        grad: np.ndarray,
        state: Dict[str, Any],
    ):
        """
        Perform one gradient descent update step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.

        lr : float
            Learning rate at current iteration.

        grad : np.ndarray
            Gradient of objective function.

        state : dict
            Optimizer state (unused in vanilla GD).

        Returns
        -------
        (np.ndarray, dict)
            Updated parameters and unchanged state.
        """

        x_new = x - lr * grad
        return x_new, state
