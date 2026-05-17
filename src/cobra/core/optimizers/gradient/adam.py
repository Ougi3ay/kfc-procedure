from __future__ import annotations

import numpy as np
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer

@OptimizerFactory.register(
    "adam",
    categories={"optimizer", "gradient"}
)
class AdamOptimizer(BaseGradientOptimizer):
    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, x, lr, grad, state):

        if "m" not in state:
            state["m"] = np.zeros_like(x)
            state["v"] = np.zeros_like(x)
            state["t"] = 0

        m, v, t = state["m"], state["v"], state["t"] + 1

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        state.update({"m": m, "v": v, "t": t})

        return x - lr * m_hat / (np.sqrt(v_hat) + 1e-8), state
