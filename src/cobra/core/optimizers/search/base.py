
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import numpy as np

from cobra.core.optimizers.base import BaseOptimizer, tqdm


class BaseSearchOptimizer(BaseOptimizer, ABC):
    """
    Base class for derivative-free optimization.
    """

    def __init__(self, show_process: bool = True):
        super().__init__(show_process=show_process)

    @abstractmethod
    def candidates(self) -> np.ndarray:
        """
        Returns:
            array shape = (n_candidates, dim)
        """
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], Any],
        init_param: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        X = self.candidates()
        n = len(X)

        iterator = tqdm(range(n), desc="search") if self.show_process else range(n)

        scores = []
        history = []

        best_score = np.inf
        best_x = None

        for i in iterator:
            x = X[i]
            score = objective(x)

            scores.append(score)

            value = np.min(score) if np.ndim(score) > 0 else score

            if value < best_score:
                best_score = value
                best_x = x.copy()
            
            history.append({
                "iter": i,
                "x": x.copy(),
                "score": score
            })

        return {
            "x": best_x,
            "score": best_score,
            "history": history,
            "scores": np.array(scores)
        }
        
