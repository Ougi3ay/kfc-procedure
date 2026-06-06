"""
Base search optimizer for COBRA framework.

This module defines derivative-free optimization strategies that
operate by evaluating a fixed or generated set of candidate solutions.

Unlike gradient-based optimizers, search optimizers:
- do not require gradients
- evaluate discrete candidate sets
- are robust for non-smooth or black-box objectives

Typical use cases in COBRA:
- kernel parameter search
- adapter selection
- hyperparameter tuning
- ensemble structure selection

Core idea:
----------
Given a candidate set X = {x_1, ..., x_n}, the optimizer evaluates
each candidate independently and selects the best according to a
risk aggregation strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any

import numpy as np

from kfc_procedure.cobra.core.optimizers.base import BaseOptimizer, tqdm


class BaseSearchOptimizer(BaseOptimizer, ABC):
    """
    Base class for derivative-free (search-based) optimizers.

    These optimizers explore a discrete candidate space instead of
    performing gradient-based updates.

    Parameters
    ----------
    show_process : bool
        Whether to display progress bar during optimization.

    risk_strategy : str
        Strategy to reduce multi-dimensional objective outputs into
        a scalar score.

        Supported:
        - "min"     : worst-case best value
        - "max"     : worst-case worst value
        - "mean"    : average score
        - "sum"     : total score
        - "median"  : robust central tendency
        - "l2"      : Euclidean norm of score vector
    """

    def __init__(self, show_process: bool = True, risk_strategy: str = "mean", **kwargs):
        super().__init__(show_process=show_process, **kwargs)
        self.risk_strategy = risk_strategy

    # Candidate generation
    @abstractmethod
    def candidates(self) -> np.ndarray:
        """
        Generate candidate solutions.

        Returns
        -------
        np.ndarray
            Array of shape (n_candidates, dim).
        """

    # Risk reduction
    def reduce_risk(self, score):
        """
        Convert possibly vector-valued score into scalar risk value.

        Parameters
        ----------
        score : float | np.ndarray
            Output from objective function.

        Returns
        -------
        float
            Reduced scalar risk.
        """

        if np.ndim(score) == 0:
            return float(score)

        score = np.asarray(score, dtype=float)

        strategies = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "l2": np.linalg.norm,
        }

        if self.risk_strategy not in strategies:
            raise ValueError(f"Unknown risk strategy: {self.risk_strategy}")

        return strategies[self.risk_strategy](score)

    # Selection
    def select_best_index(self, risks: np.ndarray) -> int:
        """
        Select index of best candidate (lowest risk).

        If multiple candidates share the same best score,
        returns the middle one for stability.
        """

        risks = np.asarray(risks)
        best = np.min(risks)

        ids = np.where(risks == best)[0]

        if len(ids) == 1:
            return int(ids[0])

        return int(ids[len(ids) // 2])

    # Optimization loop
    def optimize(
        self,
        objective: Callable[[np.ndarray], Any],
        init_param: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """
        Run search-based optimization over candidate set.

        Parameters
        ----------
        objective : callable
            Function mapping x → score (scalar or vector).

        init_param : optional
            Not used (kept for API compatibility).

        Returns
        -------
        dict
            Optimization result containing:
            - best solution
            - best score
            - full evaluation history
        """

        X = self.candidates()
        n = len(X)

        iterator = (
            tqdm(range(n), desc="search")
            if self.show_process
            else range(n)
        )

        history = []
        raw_scores = []
        reduced_scores = []

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
