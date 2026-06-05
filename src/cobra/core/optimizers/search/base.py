
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
import numpy as np

from cobra.core.optimizers.base import BaseOptimizer, tqdm


class BaseSearchOptimizer(BaseOptimizer, ABC):
    """
    Base class for derivative-free optimization.
    """

    def __init__(self, show_process: bool = True, risk_strategy: str = "min", **kwargs):
        super().__init__(show_process=show_process, **kwargs)
        self.risk_strategy = risk_strategy

    @abstractmethod
    def candidates(self) -> np.ndarray:
        """
        Returns:
            array shape = (n_candidates, dim)
        """
    
    def reduce_risk(self, score):
        if np.ndim(score) == 0:
            return float(score)
        
        score = np.asarray(score, dtype=float)

        if self.risk_strategy == "mean":
            return np.mean(score)

        elif self.risk_strategy == "sum":
            return np.sum(score)

        elif self.risk_strategy == "max":
            return np.max(score)

        elif self.risk_strategy == "min":
            return np.min(score)

        elif self.risk_strategy == "median":
            return np.median(score)

        elif self.risk_strategy == "l2":
            return np.linalg.norm(score)

        else:
            raise ValueError(
                f"Unknown risk strategy: {self.risk_strategy}"
            )
    
    def select_best_index(self, risks):
        risks = np.asarray(risks)
        best = np.min(risks)

        ids = np.where(risks == best)[0]

        if len(ids) == 1:
            return ids[0]
        
        return ids[len(ids) // 2]
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], Any],
        init_param: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        X = self.candidates()
        n = len(X)

        iterator = tqdm(range(n), desc="search") if self.show_process else range(n)

        history = []
        raw_scores = []
        reduced_scores = []

        best_score = np.inf
        best_x = None

        for i in iterator:
            x = X[i]
            score = objective(x)

            risk = self.reduce_risk(score)
            raw_scores.append(score)
            reduced_scores.append(risk)

            history.append({
                "iter": i,
                "x": x.copy(),
                "score": score,
                "risk": risk,
            })
        
        reduced_scores = np.asarray(reduced_scores)
        best_idx = self.select_best_index(reduced_scores)

        return {
            "x": X[best_idx],
            "score": raw_scores[best_idx],
            "risk": reduced_scores[best_idx],
            "best_index": best_idx,
            "history": history,
            "scores": np.array(raw_scores, dtype=object),
            "risks": reduced_scores,
        }
        
