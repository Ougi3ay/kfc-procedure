
from typing import Any, Callable, Dict

import numpy as np
from cobra.core.optimizers.base import BaseOptimizer, OptimizerFactory, tqdm


@OptimizerFactory.register("grad", "gradient_descent")
class GradientDescentOptimizer(BaseOptimizer):
    """
    Advanced Gradient Descent optimizer with adaptive step control.

    Features
    --------
    - Central difference gradient
    - Adaptive learning rate schedules
    - Smart initialization via random search
    - Gradient sign-based stabilization
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-6,
        eps=1e-6,
        grad_clip=None,
        speed="constant",
        n_tries=10,
        init_range=(1e-4, 3.0),
        show_process=True,
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

    def _gradient(self, objective, x):
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += self.eps
            x2[i] -= self.eps

            grad[i] = (objective(x1) - objective(x2)) / (2 * self.eps)

        return grad

    def _rate(self, t, base_lr):
        schedules = {
            "constant": lambda t, lr: lr,
            "linear": lambda t, lr: t * lr,
            "log": lambda t, lr: np.log1p(t) * lr,
            "sqrt": lambda t, lr: np.sqrt(1 + t) * lr,
            "quad": lambda t, lr: (1 + t**2) * lr,
            "exp": lambda t, lr: np.exp(t) * lr,
        }
        return schedules.get(self.speed, schedules["constant"])(t, base_lr)

    def _initialize(self, objective):
        low, high = self.init_range
        candidates = np.linspace(low, high, self.n_tries)

        scores = [objective(np.array([c])) for c in candidates]
        best_idx = int(np.argmin(scores))

        return np.array([candidates[best_idx]])

    def optimize(self, objective, init_param=None):
        if init_param is None:
            x = self._initialize(objective)
        else:
            x = np.array(init_param, dtype=float)

        grad = self._gradient(objective, x)
        grad_prev = grad.copy()

        base_lr = self.learning_rate / (np.linalg.norm(grad) + 1e-12)

        history = []

        iterator = range(self.max_iter)
        if self.show_process:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Gradient Descent")

        for t in iterator:
            grad = self._gradient(objective, x)
            grad_norm = np.linalg.norm(grad)

            # gradient clipping
            if self.grad_clip and grad_norm > self.grad_clip:
                grad = grad / grad_norm * self.grad_clip

            # adaptive step
            lr_t = self._rate(t, base_lr)
            x_new = x - lr_t * grad

            # prevent invalid values
            if np.any(np.isnan(x_new)) or np.any(x_new < 0):
                x_new = x * 0.95

            # sign flip detection
            if t > 3 and np.sign(grad).dot(np.sign(grad_prev)) < 0:
                base_lr *= 0.99

            loss = objective(x)

            history.append({
                "iter": t,
                "x": x.copy(),
                "loss": loss,
                "grad": grad.copy(),
                "grad_norm": grad_norm
            })

            # stopping condition
            if grad_norm < self.tol:
                break

            x = x_new
            grad_prev = grad.copy()

            if self.show_process and t % 1 == 0:
                stop_criteria = grad_norm
                iterator.set_postfix({
                    "iter": t + 1,
                    "loss": f"{loss:.6f}",
                    "x": np.round(x[0], 4) if len(x) == 1 else np.round(x, 3),
                    "grad": np.round(grad[0], 6) if len(grad) == 1 else np.round(grad, 4),
                    "stop": f"{stop_criteria:.6f}"
                })

        return {
            "x": x,
            "score": objective(x),
            "history": history
        }

@OptimizerFactory.register('adam')
class AdamOptimizer(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) Optimizer.
    """
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8,
        max_iter: int = 1000,
        tol: float = 1e-7,
        epsilon_grad: float = 1e-7,
        show_process: bool = True
    ):
        super().__init__(show_process=show_process)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.eps_grad = epsilon_grad

    def _get_numerical_gradient(self, f, x):
        """Calculates numerical gradient using central difference."""
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            old_val = x[i]
            x[i] = old_val + self.eps_grad
            fx_plus = f(x)
            x[i] = old_val - self.eps_grad
            fx_minus = f(x)
            grad[i] = (fx_plus - fx_minus) / (2 * self.eps_grad)
            x[i] = old_val  # Restore
        return grad

    def optimize(
        self, 
        objective: Callable[[np.ndarray], float], 
        init_param: np.ndarray
    ) -> Dict[str, Any]:
        
        x = init_param.astype(float).copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = []
        
        pbar = tqdm(range(1, self.max_iter + 1), disable=not self.show_process)
        
        for t in pbar:
            # 1. Compute Gradient
            grad = self._get_numerical_gradient(objective, x)
            
            # 2. Update Moving Averages
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad**2)
            
            # 3. Bias Correction
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            # 4. Update Parameters
            diff = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            x -= diff
            
            # Record keeping
            current_score = objective(x)
            history.append({"iter": t, "score": current_score, "x": x.copy()})
            
            if self.show_process:
                pbar.set_description(f"Score: {current_score:.6f}")

            # Convergence Check
            if np.linalg.norm(diff) < self.tol:
                break
                
        return {
            "x": x,
            "score": objective(x),
            "history": history
        }
