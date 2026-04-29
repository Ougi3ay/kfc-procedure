"""
Grid search optimizer for exhaustive hyperparameter exploration.

This module implements a deterministic search strategy that evaluates
all combinations of a predefined parameter grid.

Pipeline position
-----------------
Input → Splitter → Estimators → Normalize Constants → Distance
→ Kernel Adapter → Kernel → Optimize + Loss → Aggregation → Output

Purpose
-------
Grid search is used for:

- exhaustive evaluation of discrete hyperparameters
- benchmarking COBRA pipeline components
- deterministic model selection
- small to medium-sized configuration spaces

Unlike stochastic or gradient-based optimizers, grid search:

- evaluates every combination in the search space
- guarantees global optimum within the grid
- is fully deterministic
- is computationally expensive for large spaces

Method
------
Given a parameter grid:

    param_grid = {
        "k": [3, 5, 7],
        "gamma": [0.1, 0.5]
    }

Grid search evaluates:

    (3, 0.1), (3, 0.5),
    (5, 0.1), (5, 0.5),
    (7, 0.1), (7, 0.5)

and selects the best configuration according to the objective.

Design goals
------------
- exhaustive and deterministic search
- simple baseline optimizer
- easy integration with COBRA factories
- full evaluation traceability
- compatible with black-box objectives

Examples
--------
>>> optimizer = GridSearchOptimizer(param_grid={
...     "k": [3, 5, 7],
...     "gamma": [0.1, 0.5]
... })

>>> best_params, history = optimizer(objective_fn)
"""

import numpy as np
from itertools import product

from cobra.core.optimizers.search.base import (
    BaseSearchOptimizer,
    SearchOptimizerFactory,
)


@SearchOptimizerFactory.register("grid", "grid_search")
class GridSearchOptimizer(BaseSearchOptimizer):
    """
    Exhaustive grid search optimizer.

    This optimizer evaluates all combinations of hyperparameters
    defined in a parameter grid and selects the best configuration
    based on an objective function.

    Parameters
    ----------
    param_grid : dict
        Dictionary mapping parameter names to lists of values.

    verbose : bool, default=False
        If True, enables progress reporting (reserved for future use).

    Notes
    -----
    - Does not use sampling (sample() is not implemented)
    - Fully deterministic
    - Complexity grows exponentially with grid size

    Examples
    --------
    >>> optimizer = GridSearchOptimizer({
    ...     "alpha": [0.1, 1.0],
    ...     "beta": [0.01, 0.1]
    ... })
    """

    def __init__(self, param_grid: dict, verbose=False, **kwargs):
        """
        Initialize grid search optimizer.

        Parameters
        ----------
        param_grid : dict
            Search space definition.

        verbose : bool
            Whether to enable verbose output.
        """
        self.param_grid = param_grid
        self.verbose = verbose

    def search_space(self):
        """
        Return the full search space.

        Returns
        -------
        dict
            Parameter grid used for exhaustive search.
        """
        return self.param_grid

    def sample(self):
        """
        Sampling is not supported in grid search.

        Raises
        ------
        NotImplementedError
            Grid search evaluates full Cartesian product instead.
        """
        raise NotImplementedError("GridSearch does not use sampling.")

    def __call__(self, objective, *args, **kwargs):
        """
        Run exhaustive grid search.

        Parameters
        ----------
        objective : callable
            Function that evaluates a parameter vector and returns a score.

        Returns
        -------
        tuple
            (best_parameters, history)

            - best_parameters : np.ndarray
                Best parameter vector found.
            - history : list
                List of (params_dict, score) evaluations.

        Notes
        -----
        - Evaluates full Cartesian product of parameter grid
        - Tracks full evaluation history
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        best_params = None
        best_score = float("inf")
        history = []

        for combo in product(*values):
            params = dict(zip(keys, combo))

            score = objective(list(params.values()))

            history.append((params, score))

            if score < best_score:
                best_score = score
                best_params = params

        return np.array(list(best_params.values())), history
