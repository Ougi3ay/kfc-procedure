"""
Adam optimizer for COBRA framework.

This module implements the Adam (Adaptive Moment Estimation)
optimization algorithm, which combines:

- Momentum (first moment estimate)
- Adaptive learning rates (second moment estimate)

Update rules:

    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    v_t = β2 * v_{t-1} + (1 - β2) * g_t^2

Bias correction:

    m_hat = m_t / (1 - β1^t)
    v_hat = v_t / (1 - β2^t)

Parameter update:

    x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + ε)

This optimizer is widely used for:
- nonlinear COBRA loss landscapes
- kernel/adaptor optimization
- high-dimensional parameter tuning
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from kfc_procedure.cobra.core.optimizers.base import OptimizerFactory
from kfc_procedure.cobra.core.optimizers.gradient.base import BaseGradientOptimizer


@OptimizerFactory.register(
    "adam",
    categories={"optimizer", "gradient"},
)
class AdamOptimizer(BaseGradientOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    This optimizer maintains exponential moving averages of:
    - gradients (first moment)
    - squared gradients (second moment)

    Parameters
    ----------
    beta1 : float
        Decay rate for first moment (momentum).

    beta2 : float
        Decay rate for second moment (variance).

    epsilon : float
        Numerical stability constant.

    Notes
    -----
    - Robust to noisy gradients
    - Works well in non-convex optimization
    - Default choice for COBRA kernel optimization
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(
        self,
        x: np.ndarray,
        lr: float,
        grad: np.ndarray,
        state: Dict[str, Any],
    ):
        """
        Perform one Adam optimization step.

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
            - m : first moment
            - v : second moment
            - t : timestep

        Returns
        -------
        (np.ndarray, dict)
            Updated parameters and state.
        """

        if "m" not in state:
            state["m"] = np.zeros_like(x)
            state["v"] = np.zeros_like(x)
            state["t"] = 0

        m, v, t = state["m"], state["v"], state["t"] + 1

        # Moment updates
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        # Bias correction
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        # Update state
        state["m"] = m
        state["v"] = v
        state["t"] = t

        # Parameter update
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        return x_new, state