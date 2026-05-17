

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np
from cobra.core.optimizers._utils import compute_gradient
from cobra.core.optimizers.base import BaseOptimizer, tqdm


class BaseGradientOptimizer(BaseOptimizer, ABC):
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

    def gradient(
        self,
        objective: Callable,
        params: np.ndarray,
        grad_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        return compute_gradient(
            objective=objective,
            params=params,
            gradient=grad_fn,
            method=self.gradient_method,
            eps=self.eps
        )
    def _rate(self, t, lr):

        schedules = {
            'constant' : lambda x, y: y,
            'linear' : lambda x, y: x*y,
            'log' : lambda x, y: np.log(1+x) * y,
            'sqrt_root' : lambda x, y: np.sqrt(1+x) * y,
            'quad' : lambda x, y: (1+x ** 2) * y,
            'exp' : lambda x, y: np.exp(x) * y
        }
        scale = schedules.get(self.speed, schedules["constant"])(t, lr)
        return scale
    
    def _initialize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int = 1,
    ):
        low, high = self.init_range
        grid = np.linspace(low, high, self.n_tries)

        candidates = np.array([
            np.full(dim, g) for g in grid
        ])
        scores = np.array([
            objective(x)
            for x in candidates
        ])
        best_idx = np.argmin(scores)
        return candidates[best_idx].copy()
        

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        lr: float,
        grad: np.ndarray,
        state: Dict[str, Any],
    ):
        """
        One optimization step (update rule only).
        """
        pass
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param: np.ndarray | None = None,
        grad_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        if init_param is None:
            x = self._initialize(objective, dim=1)
        else:
            x = np.asarray(init_param, dtype=float)
        
        x = x.astype(float)
        state: Dict[str, Any] = {}
        
        # initail gradient
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

            if t > 3:
                if np.any(np.sign(grad_new) != np.sign(prev_grad)):
                    r0 *= 0.99
            
            # update state
            x = x_new
            prev_grad = grad_new.copy()
            grad = grad_new.copy()

            history.append({
                "iter": t+1,
                "x": x.copy(),
                "score": score,
                "grad" : grad,
                "grad_norm": float(np.linalg.norm(grad)),
                "lr": lr_t,
            })

            if self.show_process:
                iterator.set_description(
                    f"iter={t+1} | "
                    f"score={score:.4f} | "
                    f"grad={grad} | "
                    f"lr={lr_t:.6f}"
                )
            
        return {
            "x": best_x,
            "score": best_score,
            "history": history,
        }
