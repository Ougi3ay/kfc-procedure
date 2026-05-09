from __future__ import annotations
from itertools import product
from typing import Dict, List

import numpy as np

from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.search.base import BaseSearchOptimizer


@OptimizerFactory.register("grid", categories={"search"})
class GridSearchOptimizer(BaseSearchOptimizer):

    def __init__(self, param_grid: Dict[str, List[float]], **kwargs):
        super().__init__(**kwargs)
        self.param_grid = param_grid

    def candidates(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        grid = list(product(*values))

        return np.array([np.array(g, dtype=float) for g in grid])
