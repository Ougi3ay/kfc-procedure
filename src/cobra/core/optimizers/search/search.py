"""
Grid Search optimizer for COBRA framework.

This optimizer performs exhaustive search over a predefined parameter
grid. It evaluates every possible combination of hyperparameters and
selects the best configuration according to the objective function.

This is a classical derivative-free optimization strategy used for:
- kernel hyperparameter tuning
- adapter configuration search
- model selection in COBRA pipelines
- baseline comparison for advanced optimizers

Characteristics:
----------------
- deterministic (no randomness)
- exhaustive evaluation
- guaranteed best solution within grid
- computationally expensive for large grids
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List

import numpy as np

from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.optimizers.search.base import BaseSearchOptimizer


@OptimizerFactory.register(
    "grid",
    categories={"search", "derivative_free"},
)
class GridSearchOptimizer(BaseSearchOptimizer):
    """
    Grid Search optimizer.

    Parameters
    ----------
    param_grid : Dict[str, List[float]]
        Dictionary defining parameter search space.
        Example:
        {
            "alpha": [0.1, 1.0, 10.0],
            "beta": [0.01, 0.1]
        }

    Notes
    -----
    - Each combination is evaluated independently
    - Order is preserved based on itertools.product
    - Best configuration is selected using risk reduction strategy
    """

    def __init__(self, param_grid: Dict[str, List[float]], **kwargs):
        super().__init__(**kwargs)
        self.param_grid = param_grid

    # Candidate generation
    def candidates(self) -> np.ndarray:
        """
        Generate full grid of parameter combinations.

        Returns
        -------
        np.ndarray
            Shape (n_combinations, n_params)
        """

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        grid = list(product(*values))

        return np.array(
            [np.array(g, dtype=float) for g in grid],
            dtype=float,
        )
