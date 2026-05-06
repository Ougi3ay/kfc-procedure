from itertools import product

import numpy as np

from cobra.core.optimizers.base import BaseOptimizer, OptimizerFactory, tqdm


@OptimizerFactory.register("grid", "grid_search")
class GridSearchOptimizer(BaseOptimizer):
    """
    Exhaustive grid search optimizer.

    Suitable for discrete parameter spaces.

    Parameters
    ----------
    param_grid : dict
        Dictionary of parameter lists
    max_evals : int, optional
        Limit number of evaluations
    """

    def __init__(
        self,
        param_grid: dict,
        max_evals: int | None = None,
        show_process: bool = True,
        **kwargs
    ):
        super().__init__(show_process=show_process, **kwargs)
        self.param_grid = param_grid
        self.max_evals = max_evals

    def optimize(self, objective, init_param=None):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combos = list(product(*values))
        total = len(combos)

        best_x = None
        best_score = float("inf")
        history = []

        iterator = combos
        if self.show_process:
            iterator = tqdm(combos, total=total, desc="Grid Search")

        for i, combo in enumerate(iterator):
            if self.max_evals and i >= self.max_evals:
                break

            params = dict(zip(keys, combo))
            x = np.array([params[k] for k in keys])

            score = objective(x)

            history.append({
                "iter": i,
                "x": x.copy(),
                "params": params,
                "score": score
            })

            if score < best_score:
                best_score = score
                best_x = x.copy()

            if self.show_process and i % 10 == 0:
                iterator.set_postfix({"best": f"{best_score:.4f}"})

        return {
            "x": best_x,
            "score": best_score,
            "history": history
        }
