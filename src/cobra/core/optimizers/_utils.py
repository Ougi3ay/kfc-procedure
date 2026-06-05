"""
Numerical gradient computation utilities for COBRA framework.

This module provides multiple finite-difference and stochastic
gradient estimation methods used in COBRA optimization components.

These gradients are used when:
- analytic gradients are unavailable
- black-box objective functions are used
- kernel/adaptor/loss functions are non-differentiable
- search space is complex or noisy

Supported methods:
------------------
- Central difference (most stable default)
- Forward difference (cheaper, less accurate)
- SPSA (stochastic approximation)
- Complex-step (high precision, requires analytic compatibility)
- Parallel central difference (CPU-accelerated)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


# ============================================================
# Central Difference Gradient
# ============================================================
def central_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute gradient using central difference approximation.

    Formula:
        g_i = (f(x + eps) - f(x - eps)) / (2 * eps)

    This is the most stable numerical method.
    """

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


# ============================================================
# Forward Difference Gradient
# ============================================================
def forward_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute gradient using forward difference approximation.

    Formula:
        g_i = (f(x + eps) - f(x)) / eps
    """

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


# ============================================================
# SPSA Gradient (Stochastic Approximation)
# ============================================================
def spsa_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA).

    Efficient gradient estimation using random perturbation vector.

    Formula:
        g ≈ (f(x + eps * Δ) - f(x - eps * Δ)) / (2 * eps * Δ)
    """

    p = np.asarray(params, dtype=float).copy()

    delta = np.random.choice([-1.0, 1.0], size=p.shape)

    f_plus = objective(p + eps * delta)
    f_minus = objective(p - eps * delta)

    return (f_plus - f_minus) / (2 * eps * delta)


# ============================================================
# Complex Step Gradient
# ============================================================
def complex_step_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-20,
) -> np.ndarray:
    """
    High-precision gradient using complex-step differentiation.

    Formula:
        g_i = Im(f(x + i eps)) / eps

    Requires objective to support complex numbers.
    """

    p = np.asarray(params, dtype=np.complex128)
    grad = np.empty(p.shape, dtype=float)

    for i in range(p.size):
        z = p.copy()
        z[i] += 1j * eps

        grad[i] = np.imag(objective(z)) / eps

    return grad


# ============================================================
# Parallel Central Difference
# ============================================================
def parallel_central_difference_gradient(
    objective: Callable,
    params: np.ndarray,
    eps: float = 1e-7,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Parallelized central difference gradient computation.

    Useful for expensive objective functions.
    """

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
            delayed(compute_i)(i) for i in range(p.size)
        )
    )


# ============================================================
# Dispatcher
# ============================================================
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
) -> np.ndarray:
    """
    Unified gradient computation interface.

    Parameters
    ----------
    objective : callable
        Function f(x) → scalar

    params : np.ndarray
        Input parameters

    gradient : callable, optional
        If provided, overrides numerical methods

    method : str
        Gradient method to use

    eps : float
        Finite difference step size

    n_jobs : int or None
        Parallel workers (only for parallel method)

    Returns
    -------
    np.ndarray
        Gradient vector
    """

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

    return GRADIENT_METHODS[method](objective, params, eps)