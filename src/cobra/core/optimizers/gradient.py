
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

    # -------------------------------------------------
    # Gradient (central difference)
    # -------------------------------------------------
    def _gradient(self, objective, x):
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += self.eps
            x2[i] -= self.eps

            grad[i] = (objective(x1) - objective(x2)) / (2 * self.eps)

        return grad

    # -------------------------------------------------
    # Learning rate schedules (from your code)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Smart initialization (from your code)
    # -------------------------------------------------
    def _initialize(self, objective):
        low, high = self.init_range
        candidates = np.linspace(low, high, self.n_tries)

        scores = [objective(np.array([c])) for c in candidates]
        best_idx = int(np.argmin(scores))

        return np.array([candidates[best_idx]])

    def optimize(self, objective, init_param=None):
        # ---- Initialization ----
        if init_param is None:
            x = self._initialize(objective)
        else:
            x = np.array(init_param, dtype=float)

        grad = self._gradient(objective, x)
        grad_prev = grad.copy()

        # normalize first step (your idea)
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

            # prevent invalid values (your logic)
            if np.any(np.isnan(x_new)) or np.any(x_new < 0):
                x_new = x * 0.95

            # sign flip detection (your idea)
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

            if self.show_process and t % 5 == 0:
                iterator.set_postfix({
                    "loss": f"{loss:.4f}",
                    "grad": f"{grad_norm:.4f}"
                })

        return {
            "x": x,
            "score": objective(x),
            "history": history
        }
