"""
Gradient-based optimizer module for COBRA framework.

This module defines the base class for all gradient-based optimization
strategies used in COBRA.

Gradient optimizers are responsible for solving:

    argmin_x  f(x)

using iterative updates based on gradient information.

This class supports:
- adaptive learning rate schedules
- finite-difference or analytical gradients
- initialization heuristics
- early stopping
- optimization history tracking

It is used in:
- kernel parameter tuning
- adapter weight learning
- distance metric calibration
- COBRA aggregation optimization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

from cobra.core.optimizers._utils import compute_gradient
from cobra.core.optimizers.base import BaseOptimizer, tqdm


class BaseGradientOptimizer(BaseOptimizer, ABC):
    """
    Base class for gradient-based optimization algorithms.

    This class implements the core optimization loop and delegates
    the update rule to subclasses via the `step()` method.

    Parameters
    ----------
    learning_rate : float
        Initial learning rate.

    max_iter : int
        Maximum number of optimization iterations.

    tol : float
        Stopping tolerance for gradient norm.

    speed : str
        Learning rate schedule:
        - constant
        - linear
        - log
        - sqrt_root
        - quad
        - exp

    gradient_method : str
        Gradient computation method:
        - forward
        - backward
        - central (default finite difference)

    eps : float
        Small epsilon used for numerical gradient estimation.

    n_tries : int
        Number of initialization candidates.

    init_range : tuple
        Range for random initialization search.

    show_process : bool
        Whether to display progress bar.

    Attributes
    ----------
    learning_rate : float
    gradient_method : str
    eps : float
    tol : float
    n_tries : int
    init_range : tuple
    speed : str
    max_iter : int
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 300,
        tol: float = 1e-7,
        speed: str = "constant",
        gradient_method: str = "central",
        eps: float = 1e-7,
        n_tries: int = 5,
        init_range=(1e-4, 3.0),
        show_process: bool = True,
        **kwargs,
    ):
        super().__init__(show_process=show_process, **kwargs)

        self.learning_rate = learning_rate
        self.gradient_method = gradient_method
        self.eps = eps
        self.tol = tol
        self.n_tries = n_tries
        self.init_range = init_range
        self.speed = speed
        self.max_iter = max_iter

    # =========================================================
    # Gradient computation
    # =========================================================
    def gradient(
        self,
        objective: Callable,
        params: np.ndarray,
        grad_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Compute gradient of objective function.

        Supports:
        - analytical gradient (if provided)
        - numerical finite-difference approximation
        """
        return compute_gradient(
            objective=objective,
            params=params,
            gradient=grad_fn,
            method=self.gradient_method,
            eps=self.eps,
        )

    # =========================================================
    # Learning rate schedule
    # =========================================================
    def _rate(self, t: int, lr: float) -> float:
        """
        Compute learning rate schedule.

        Parameters
        ----------
        t : int
            Iteration step.

        lr : float
            Base learning rate.

        Returns
        -------
        float
            Scaled learning rate.
        """

        schedules = {
            "constant": lambda x, y: y,
            "linear": lambda x, y: x * y,
            "log": lambda x, y: np.log(1 + x) * y,
            "sqrt_root": lambda x, y: np.sqrt(1 + x) * y,
            "quad": lambda x, y: (1 + x**2) * y,
            "exp": lambda x, y: np.exp(x) * y,
        }

        return schedules.get(self.speed, schedules["constant"])(t, lr)

    # =========================================================
    # Initialization
    # =========================================================
    def _initialize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int = 1,
    ) -> np.ndarray:
        """
        Initialize parameters using coarse grid search.

        This avoids poor local minima at initialization.
        """
        low, high = self.init_range
        grid = np.linspace(low, high, self.n_tries)

        candidates = np.array([np.full(dim, g) for g in grid])

        scores = np.array([objective(x) for x in candidates])

        best_idx = np.argmin(scores)

        return candidates[best_idx].copy()

    # =========================================================
    # Update rule (to be implemented by subclasses)
    # =========================================================
    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        lr: float,
        grad: np.ndarray,
        state: Dict[str, Any],
    ):
        """
        Perform one optimization step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.

        lr : float
            Learning rate.

        grad : np.ndarray
            Gradient of objective.

        state : dict
            Optimizer internal state (momentum, cache, etc.)

        Returns
        -------
        (np.ndarray, dict)
            Updated parameters and updated state.
        """
        pass

    # =========================================================
    # Main optimization loop
    # =========================================================
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param: np.ndarray | None = None,
        grad_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run gradient-based optimization.

        Returns best solution found along with training history.
        """

        if init_param is None:
            x = self._initialize(objective, dim=1)
        else:
            x = np.asarray(init_param, dtype=float)

        x = x.astype(float)
        state: Dict[str, Any] = {}

        grad = self.gradient(objective, x, grad_fn)
        prev_grad = grad.copy()

        r0 = self.learning_rate / (np.linalg.norm(grad) + 1e-12)

        best_x = x.copy()
        best_score = objective(x)

        history = []

        iterator = (
            tqdm(range(self.max_iter), desc="GD", leave=True)
            if self.show_process
            else range(self.max_iter)
        )

        for t in iterator:
            lr_t = self._rate(t, r0)

            x_new, state = self.step(x, lr_t, grad, state)

            if np.any(np.isnan(x_new)):
                x_new = x * 0.95

            grad_new = self.gradient(objective, x_new, grad_fn)
            score = objective(x_new)

            if score < best_score:
                best_score = score
                best_x = x_new.copy()

            if np.linalg.norm(grad_new) < self.tol:
                x = x_new
                break

            if t > 3 and np.any(np.sign(grad_new) != np.sign(prev_grad)):
                r0 *= 0.99

            x = x_new
            prev_grad = grad_new.copy()
            grad = grad_new.copy()

            history.append(
                {
                    "iter": t + 1,
                    "x": x.copy(),
                    "score": score,
                    "grad": grad,
                    "grad_norm": float(np.linalg.norm(grad)),
                    "lr": lr_t,
                }
            )

            if self.show_process:
                iterator.set_description(
                    f"iter={t+1} | score={score:.4f} | grad_norm={np.linalg.norm(grad):.4f}"
                )

        return {
            "x": best_x,
            "score": best_score,
            "history": history,
        }
