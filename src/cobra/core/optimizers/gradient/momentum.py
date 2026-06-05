"""
Momentum optimizer for COBRA framework.

This module implements classical Momentum-based gradient descent,
which accelerates optimization by accumulating a velocity vector
in the direction of consistent gradients.

Update rules:

    v_t = μ * v_{t-1} - lr * g_t
    x_{t+1} = x_t + v_t

Where:
- v is the velocity (momentum term)
- μ is the momentum coefficient
- g_t is the gradient

This optimizer helps:
- reduce oscillations in steep directions
- accelerate convergence in shallow valleys
- stabilize COBRA kernel/adaptor optimization
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer


@OptimizerFactory.register(
    "momentum",
    categories={"optimizer", "gradient"},
)
class MomentumOptimizer(BaseGradientOptimizer):
    """
    Momentum-based gradient descent optimizer.

    This optimizer extends vanilla gradient descent by introducing
    a velocity term that accumulates past gradients.

    Parameters
    ----------
    momentum : float
        Momentum coefficient (μ). Typically in [0.8, 0.99].

    Notes
    -----
    - Helps escape shallow local minima
    - Reduces zig-zag behavior in optimization paths
    - Commonly used in COBRA kernel learning
    """

    def __init__(self, momentum: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def step(
        self,
        x: np.ndarray,
        lr: float,
        grad: np.ndarray,
        state: Dict[str, Any],
    ):
        """
        Perform one momentum update step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.

        lr : float
            Learning rate.

        grad : np.ndarray
            Gradient of objective.

        state : dict
            Internal optimizer state:
            - v : velocity vector

        Returns
        -------
        (np.ndarray, dict)
            Updated parameters and state.
        """

        if "v" not in state:
            state["v"] = np.zeros_like(x)

        v = state["v"]

        # velocity update
        v = self.momentum * v - lr * grad

        state["v"] = v

        # parameter update
        x_new = x + v

        return x_new, state
