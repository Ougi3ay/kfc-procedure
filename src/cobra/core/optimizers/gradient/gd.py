
from __future__ import annotations
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer


@OptimizerFactory.register(
    "gd",
    categories={"optimizer", "gradient"}
)
class GradientDescentOptimizer(BaseGradientOptimizer):
    def step(self, x, lr, grad, state):
        return x - lr * grad, state

