import numpy as np
import pytest

from cobra.core.optimizers import OptimizerFactory
from cobra.core.optimizers.gradient import (
    GradientDescentOptimizer,
    MomentumOptimizer,
    AdamOptimizer,
)
from cobra.core.optimizers.search import GridSearchOptimizer


# ============================================================
# Test objective (simple convex quadratic)
# ============================================================
def quadratic(x):
    return np.sum(x ** 2)


# ============================================================
# Gradient optimizers
# ============================================================
@pytest.mark.parametrize("opt_cls", [
    GradientDescentOptimizer,
    MomentumOptimizer,
    AdamOptimizer,
])
def test_gradient_optimizers_reduce_loss(opt_cls):
    opt = opt_cls(max_iter=50, show_process=False, learning_rate=0.1)

    result = opt.optimize(quadratic, init_param=np.array([5.0]))

    x = result["x"]
    score = result["score"]

    # should move toward 0
    assert abs(x[0]) < 5.0
    assert score < quadratic(np.array([5.0]))


def test_gradient_shapes():
    opt = GradientDescentOptimizer(max_iter=10, show_process=False)

    result = opt.optimize(quadratic, init_param=np.array([3.0, -4.0]))

    assert result["x"].shape == (2,)
    assert np.isfinite(result["score"])


# ============================================================
# Momentum specific behavior
# ============================================================
def test_momentum_smooth_behavior():
    opt = MomentumOptimizer(
        max_iter=20,
        learning_rate=0.1,
        show_process=False,
        momentum=0.9,
    )

    result = opt.optimize(quadratic, init_param=np.array([10.0]))

    assert result["score"] < quadratic(np.array([10.0]))


# ============================================================
# Adam optimizer stability
# ============================================================
def test_adam_convergence():
    opt = AdamOptimizer(
        max_iter=50,
        learning_rate=0.05,
        show_process=False,
    )

    result = opt.optimize(quadratic, init_param=np.array([8.0]))

    assert np.isfinite(result["score"])
    assert abs(result["x"][0]) < 8.0


# ============================================================
# Grid search optimizer
# ============================================================
def test_grid_search_optimizer():
    grid = {
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 2.0],
    }

    opt = GridSearchOptimizer(param_grid=grid, show_process=False)

    def objective(v):
        return np.sum(v ** 2)

    result = opt.optimize(objective)

    assert "x" in result
    assert "score" in result
    assert "risks" in result
    assert result["x"] is not None


# ============================================================
# Factory tests
# ============================================================
def test_optimizer_factory_create():
    opt = OptimizerFactory.create(
        "gd",
        max_iter=10,
        learning_rate=0.1,
    )

    assert isinstance(opt, GradientDescentOptimizer)


def test_optimizer_factory_register_search():
    opt = OptimizerFactory.create(
        "grid",
        param_grid={"x": [0.0, 1.0]},
    )

    assert isinstance(opt, GridSearchOptimizer)


def test_optimizer_factory_available():
    names = OptimizerFactory.available()

    assert "gd" in names or "grid" in names


def test_factory_case_insensitive():
    opt1 = OptimizerFactory.create("GD", max_iter=5)
    opt2 = OptimizerFactory.create("gd", max_iter=5)

    assert type(opt1) == type(opt2)


# ============================================================
# Sanity: optimizer returns valid history
# ============================================================
def test_optimizer_history_exists():
    opt = GradientDescentOptimizer(max_iter=10, show_process=False)

    result = opt.optimize(quadratic, init_param=np.array([3.0]))

    assert isinstance(result["history"], list)
    assert len(result["history"]) > 0
    