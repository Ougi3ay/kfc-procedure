from __future__ import annotations

import numpy as np
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer


@OptimizerFactory.register("momentum", categories={"optimizer", "gradient"})
class MomentumOptimizer(BaseGradientOptimizer):

    def __init__(self, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def step(self, x, lr, grad, state):

        if "v" not in state:
            state["v"] = np.zeros_like(x)

        v = state["v"]
        v = self.momentum * v - lr * grad

        state["v"] = v

        return x + v, state
