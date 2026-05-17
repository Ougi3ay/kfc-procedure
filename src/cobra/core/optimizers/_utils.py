

from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np


def central_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    grad = np.empty_like(p)

    for i in range(p.size):
        original = p[i]

        p[i] = original + eps
        f_plus = objective(p)

        p[i] = original - eps
        f_minus = objective(p)

        p[i] = original

        grad[i] = (f_plus - f_minus) / (2 * eps)

    return grad

def forward_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    grad = np.empty_like(p)

    f0 = objective(p)
    for i in range(p.size):
        original = p[i]

        p[i] = original + eps
        f_plus = objective(p)

        p[i] = original

        grad[i] = (f_plus - f0) / eps
    
    return grad

def spsa_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()

    delta = np.random.choice(
        [-1.0, 1.0],
        size=p.shape,
    )

    f_plus = objective(p + eps * delta)
    f_minus = objective(p - eps * delta)

    return (f_plus - f_minus) / (2 * eps * delta)

def complex_step_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-20,
) -> np.ndarray:
    p = np.asarray(params, dtype=np.complex128)
    grad = np.empty(p.shape, dtype=float)

    for i in range(p.size):
        z = p.copy()
        z[i] += 1j * eps

        grad[i] = np.imag(objective(z)) / eps

    return grad

def parallel_central_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
    n_jobs: int = -1,
) -> np.ndarray:
    from joblib import Parallel, delayed

    p = np.asarray(params, dtype=float)

    def compute_i(i: int):
        x = p.copy()

        x[i] += eps
        f_plus = objective(x)

        x[i] -= 2 * eps
        f_minus = objective(x)

        return (f_plus - f_minus) / (2 * eps)
    
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(compute_i)(i)
            for i in range(p.size)
        )
    )



# dispatcher
GRADIENT_METHODS: Dict[str, Callable] = {
    "central": central_difference_gradient,
    "forward": forward_difference_gradient,
    "spsa": spsa_gradient,
    "complex": complex_step_gradient,
    "parallel": parallel_central_difference_gradient,
}

def compute_gradient(
    objective: Callable,
    params: np.ndarray,
    gradient: Optional[Callable] = None,
    method: str = "central",
    eps: float = 1e-7,
    n_jobs: Optional[int] = None,
):
    if gradient is not None:
        return np.asarray(gradient(params), dtype=float)
    
    if method == "parallel":
        return parallel_central_difference_gradient(
            objective,
            params,
            eps,
            n_jobs=-1 if n_jobs is None else n_jobs,
        )
    
    if method not in GRADIENT_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {list(GRADIENT_METHODS.keys())}"
        )
    
    return GRADIENT_METHODS[method](
        objective,
        params,
        eps,
    )
