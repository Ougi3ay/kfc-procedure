import numpy as np
import pytest

from kfc_procedure.cobra.core.losses import LossFactory
from kfc_procedure.cobra.core.losses import (
    MSELoss,
    MAELoss,
    HuberLoss,
    LogLoss,
    HingeLoss,
    QuantileLoss,
)


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def regression_data():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])
    return y_true, y_pred


@pytest.fixture
def classification_data():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    return y_true, y_pred


# =========================================================
# Factory tests
# =========================================================
def test_loss_factory_create():
    loss = LossFactory.create("mse")
    assert isinstance(loss, MSELoss)


def test_loss_factory_case_insensitive():
    l1 = LossFactory.create("MSE")
    l2 = LossFactory.create("mse")

    assert type(l1) == type(l2)


def test_loss_factory_available():
    names = LossFactory.available()
    assert "mse" in names
    assert "mae" in names
    assert "huber" in names


# =========================================================
# MSE
# =========================================================
def test_mse_loss(regression_data):
    y_true, y_pred = regression_data

    loss = MSELoss()
    result = loss(y_true, y_pred)

    expected = np.mean((y_true - y_pred) ** 2)

    assert np.isclose(result, expected)


# =========================================================
# MAE
# =========================================================
def test_mae_loss(regression_data):
    y_true, y_pred = regression_data

    loss = MAELoss()
    result = loss(y_true, y_pred)

    expected = np.mean(np.abs(y_true - y_pred))

    assert np.isclose(result, expected)


# =========================================================
# Huber Loss
# =========================================================
def test_huber_loss_basic():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 2.0, 2.5])

    loss = HuberLoss(delta=1.0)
    result = loss(y_true, y_pred)

    assert result >= 0.0
    assert np.isfinite(result)


def test_huber_parameter_effect():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([0.0, 0.0])

    l1 = HuberLoss(delta=0.5)(y_true, y_pred)
    l2 = HuberLoss(delta=5.0)(y_true, y_pred)

    # larger delta should behave more like MSE → larger penalty
    assert l2 >= l1


# =========================================================
# Log Loss (classification)
# =========================================================
def test_log_loss(classification_data):
    y_true, y_pred = classification_data

    loss = LogLoss()
    result = loss(y_true, y_pred)

    assert result > 0.0
    assert np.isfinite(result)


def test_log_loss_clipping():
    y_true = np.array([1, 0])
    y_pred = np.array([0.0, 1.0])  # extreme probabilities

    loss = LogLoss()
    result = loss(y_true, y_pred)

    assert np.isfinite(result)


# =========================================================
# Hinge Loss
# =========================================================
def test_hinge_loss():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([0.8, -0.2, -0.5, 0.3])

    loss = HingeLoss()
    result = loss(y_true, y_pred)

    assert result >= 0.0
    assert np.isfinite(result)


# =========================================================
# Quantile Loss
# =========================================================
def test_quantile_loss():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 2.5])

    loss = QuantileLoss(tau=0.5)
    result = loss(y_true, y_pred)

    assert np.isfinite(result)


def test_quantile_parameter_effect():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([0.0, 0.0])

    low_tau = QuantileLoss(tau=0.1)(y_true, y_pred)
    high_tau = QuantileLoss(tau=0.9)(y_true, y_pred)

    assert np.isfinite(low_tau)
    assert np.isfinite(high_tau)


# =========================================================
# Stability / edge cases
# =========================================================
def test_loss_shape_safety():
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[1.0], [2.0]])

    loss = MSELoss()
    result = loss(y_true, y_pred)

    assert np.isfinite(result)


def test_loss_consistency_repeatability():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.2, 1.8, 3.1])

    l1 = MSELoss()(y_true, y_pred)
    l2 = MSELoss()(y_true, y_pred)

    assert np.isclose(l1, l2)
