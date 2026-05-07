from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from cobra.core.optimizers.base import (
    BaseOptimizer,
    OptimizerFactory,
    tqdm,
)


@OptimizerFactory.register("grad", "gradient_descent")
class GradientDescentOptimizer(BaseOptimizer):
    """
    Stable numerical gradient descent optimizer.

    Features
    --------
    - Central difference gradient
    - Adaptive learning-rate schedules
    - Smart initialization
    - Gradient clipping
    - Loss rollback protection
    - Best parameter tracking
    - Multi-dimensional optimization
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-6,
        grad_clip: float | None = None,
        speed: str = "constant",
        n_tries: int = 10,
        init_range=(1e-4, 3.0),
        min_value: float = 1e-12,
        patience: int = 10,
        show_process: bool = True,
        **kwargs,
    ):
        super().__init__(show_process=show_process, **kwargs)

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.grad_clip = grad_clip
        self.speed = speed
        self.n_tries = n_tries
        self.init_range = init_range
        self.min_value = min_value
        self.patience = patience

    # ==========================================================
    # Numerical Gradient
    # ==========================================================
    def _gradient(
        self,
        objective: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> np.ndarray:

        grad = np.zeros_like(x, dtype=float)

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()

            x1[i] += self.eps
            x2[i] -= self.eps

            f1 = objective(x1)
            f2 = objective(x2)

            grad[i] = (f1 - f2) / (2 * self.eps)

        return grad

    # ==========================================================
    # Learning Rate Schedules
    # ==========================================================
    def _rate(self, t: int) -> float:

        schedules = {
            "constant": lambda t: 1.0,
            "linear": lambda t: 1.0 / (1.0 + t),
            "log": lambda t: 1.0 / np.log1p(t + 1),
            "sqrt": lambda t: 1.0 / np.sqrt(t + 1),
            "quad": lambda t: 1.0 / (1.0 + t**2),
            "exp": lambda t: np.exp(-0.01 * t),
        }

        scale = schedules.get(self.speed, schedules["constant"])(t)

        return self.learning_rate * scale

    # ==========================================================
    # Smart Initialization
    # ==========================================================
    def _initialize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int = 1,
    ) -> np.ndarray:

        low, high = self.init_range

        candidates = np.linspace(low, high, self.n_tries)

        best_x = None
        best_score = np.inf

        for c in candidates:

            x = np.full(dim, c, dtype=float)

            score = objective(x)

            if np.isnan(score):
                continue

            if score < best_score:
                best_score = score
                best_x = x

        if best_x is None:
            best_x = np.full(dim, 1.0)

        return best_x

    # ==========================================================
    # Optimization
    # ==========================================================
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param=None,
    ) -> Dict[str, Any]:

        if init_param is None:
            x = self._initialize(objective)
        else:
            x = np.asarray(init_param, dtype=float)

        best_x = x.copy()
        best_score = objective(x)

        no_improve_count = 0

        history = []

        iterator = range(self.max_iter)

        if self.show_process:
            iterator = tqdm(iterator, desc="Gradient Descent")

        for t in iterator:

            grad = self._gradient(objective, x)

            grad_norm = np.linalg.norm(grad)

            # --------------------------------------
            # Gradient clipping
            # --------------------------------------
            if (
                self.grad_clip is not None
                and grad_norm > self.grad_clip
            ):
                grad = (
                    grad / grad_norm
                ) * self.grad_clip

                grad_norm = np.linalg.norm(grad)

            # --------------------------------------
            # Convergence
            # --------------------------------------
            if grad_norm < self.tol:
                break

            # --------------------------------------
            # Learning rate
            # --------------------------------------
            lr_t = self._rate(t)

            # --------------------------------------
            # Parameter update
            # --------------------------------------
            x_new = x - lr_t * grad

            # avoid invalid bandwidths
            x_new = np.maximum(
                x_new,
                self.min_value,
            )

            # --------------------------------------
            # Evaluate
            # --------------------------------------
            current_score = objective(x)
            new_score = objective(x_new)

            # --------------------------------------
            # Rollback protection
            # --------------------------------------
            if (
                np.isnan(new_score)
                or np.isinf(new_score)
            ):
                lr_t *= 0.5
                x_new = x - lr_t * grad
                new_score = objective(x_new)

            # --------------------------------------
            # Accept only improvement
            # --------------------------------------
            if new_score <= current_score:
                x = x_new
                current_score = new_score
            else:
                lr_t *= 0.5

            # --------------------------------------
            # Track best
            # --------------------------------------
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # --------------------------------------
            # Early stopping
            # --------------------------------------
            if no_improve_count >= self.patience:
                break

            # --------------------------------------
            # History
            # --------------------------------------
            history.append({
                "iter": t,
                "x": x.copy().tolist(),
                "score": float(current_score),
                "grad_norm": float(grad_norm),
                "lr": float(lr_t),
            })

            # --------------------------------------
            # Progress bar
            # --------------------------------------
            if self.show_process:
                iterator.set_postfix({
                    "score": f"{current_score:.6f}",
                    "grad": f"{grad_norm:.6f}",
                    "lr": f"{lr_t:.6f}",
                })

        return {
            "x": best_x,
            "score": best_score,
            "history": history,
        }

@OptimizerFactory.register("adam")
class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer using numerical gradients.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 1000,
        tol: float = 1e-6,
        eps_grad: float = 1e-6,
        n_tries: int = 10,
        init_range: tuple[float, float] = (1e-4, 3.0),
        show_process: bool = True,
        **kwargs,
    ):
        super().__init__(
            show_process=show_process,
            **kwargs
        )

        self.learning_rate = learning_rate

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon
        self.eps_grad = eps_grad

        self.max_iter = max_iter
        self.tol = tol

        self.n_tries = n_tries
        self.init_range = init_range

    def _gradient(self, objective, x):
        grad = np.zeros_like(x, dtype=float)

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()

            x1[i] += self.eps_grad
            x2[i] -= self.eps_grad

            grad[i] = (
                objective(x1) - objective(x2)
            ) / (2 * self.eps_grad)

        return grad

    def _initialize(self, objective):
        low, high = self.init_range

        candidates = np.linspace(
            low,
            high,
            self.n_tries
        )

        scores = [
            objective(np.array([c]))
            for c in candidates
        ]

        best_idx = int(np.argmin(scores))

        return np.array([candidates[best_idx]])

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        init_param=None,
    ) -> Dict[str, Any]:

        if init_param is None:
            x = self._initialize(objective)
        else:
            x = np.asarray(init_param, dtype=float)

        m = np.zeros_like(x)
        v = np.zeros_like(x)

        history = []

        iterator = range(1, self.max_iter + 1)

        if self.show_process:
            iterator = tqdm(
                iterator,
                desc="Adam"
            )

        for t in iterator:

            grad = self._gradient(objective, x)

            m = (
                self.beta1 * m
                + (1 - self.beta1) * grad
            )

            v = (
                self.beta2 * v
                + (1 - self.beta2) * (grad ** 2)
            )

            m_hat = m / (1 - self.beta1 ** t)

            v_hat = v / (1 - self.beta2 ** t)

            step = (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )

            x_new = x - step

            if (
                np.any(np.isnan(x_new))
                or np.any(x_new <= 0)
            ):
                x_new = np.maximum(
                    x * 0.95,
                    1e-8
                )

            loss = objective(x)

            history.append({
                "iter": t,
                "x": x.copy(),
                "loss": loss,
                "grad_norm": np.linalg.norm(grad),
            })

            if np.linalg.norm(step) < self.tol:
                break

            x = x_new

            if self.show_process:
                iterator.set_postfix({
                    "loss": f"{loss:.6f}",
                    "x": np.round(x[0], 6),
                })

        return {
            "x": x,
            "score": objective(x),
            "history": history,
        }