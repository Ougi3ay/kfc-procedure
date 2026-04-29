"""
Gradient Descent optimizer for COBRA continuous parameter tuning.

This module implements a simple finite-difference gradient descent
optimizer used to minimize black-box objective functions in the
COBRA pipeline.

Pipeline position
-----------------
Input -> Splitter -> Estimators -> Normalize Constants -> Distance
-> Kernel Adapter -> Kernel -> Optimize + Loss -> Aggregation -> Output

Purpose
-------
This optimizer performs iterative continuous optimization over model
parameters such as:

- kernel hyperparameters (e.g., gamma, bandwidth)
- distance scaling factors
- adapter weights (alpha, beta)
- smooth weighting coefficients

It uses numerical approximation of gradients via finite differences
to update parameters in the direction of decreasing loss.

Method
------
The gradient is estimated using:

- finite difference approximation (epsilon perturbation)
- iterative parameter updates
- optional convergence tolerance stopping

Design goals
------------
- simple gradient-based optimization baseline
- no dependency on automatic differentiation
- compatible with arbitrary black-box objectives
- traceable optimization history
- optional verbose progress tracking

Examples
--------
>>> optimizer = GradientDescentOptimizer(lr=0.01, max_iter=100)
>>> best_params, history = optimizer(objective_fn, init_params)
"""

from __future__ import annotations

import numpy as np

from cobra.core.optimizers.base import tqdm, trange
from cobra.core.optimizers.gradient.base import (
    BaseGradientOptimizer,
    GradientOptimizerFactory,
)


@GradientOptimizerFactory.register("grad", "gradient_descent")
class GradientDescentOptimizer(BaseGradientOptimizer):
    """
    Finite-difference Gradient Descent optimizer.

    This optimizer numerically approximates gradients and performs
    iterative updates to minimize a scalar objective function.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for parameter updates.

    max_iter : int, default=100
        Maximum number of optimization iterations.

    tol : float, default=1e-6
        Gradient norm tolerance for early stopping.

    eps : float, default=1e-8
        Finite difference step size for gradient estimation.

    verbose : bool, default=False
        If True, displays progress using tqdm.

    Notes
    -----
    This optimizer is intended for:

    - black-box optimization
    - COBRA kernel tuning
    - small-scale continuous parameter search

    It is not efficient for high-dimensional problems due to
    finite-difference gradient computation.

    Examples
    --------
    >>> optimizer = GradientDescentOptimizer(learning_rate=0.01)
    >>> params, history = optimizer(objective, init_params)
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-6,
        eps=1e-8,
        verbose=False,
        **kwargs,
    ):
        """
        Initialize gradient descent optimizer.

        Parameters
        ----------
        learning_rate : float
            Step size for updates.

        max_iter : int
            Maximum iterations.

        tol : float
            Convergence tolerance on gradient norm.

        eps : float
            Finite difference epsilon.

        verbose : bool
            Whether to display progress.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.verbose = verbose

    def gradient(self, objective, params):
        """
        Compute numerical gradient via finite differences.

        Parameters
        ----------
        objective : callable
            Objective function to minimize.

        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        np.ndarray
            Approximated gradient vector.
        """
        grad = np.zeros_like(params)

        base_value = objective(params)

        for i in range(len(params)):
            params_eps = np.array(params, copy=True)
            params_eps[i] += self.eps

            grad[i] = (objective(params_eps) - base_value) / self.eps

        return grad

    def step(self, objective, params):
        """
        Perform one gradient descent update step.

        Parameters
        ----------
        objective : callable
            Objective function.

        params : np.ndarray
            Current parameters.

        Returns
        -------
        np.ndarray
            Updated parameters.
        """
        grad = self.gradient(objective, params)
        return params - self.learning_rate * grad

    def __call__(self, objective, params):
        """
        Run full optimization loop.

        Parameters
        ----------
        objective : callable
            Function to minimize.

        params : array-like
            Initial parameter vector.

        Returns
        -------
        tuple
            (optimized_params, gradient_history)

        Notes
        -----
        Stops early if gradient norm falls below tolerance.
        """
        params = np.array(params, dtype=float)
        history = []

        iterator = (
            range(self.max_iter)
            if not self.verbose
            else tqdm(range(self.max_iter), desc="GD")
        )

        for i in iterator:
            grad = self.gradient(objective, params)
            params = self.step(objective, params)

            history.append(grad.copy())

            if np.linalg.norm(grad) < self.tol:
                break

            if self.verbose:
                iterator.set_description(
                    f"GD iter: {i} | grad norm: {np.linalg.norm(grad):.4f}"
                )

        return params, history
