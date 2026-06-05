import numpy as np
import pytest

from cobra.core.estimators import (
    MeanRegressor,
    EstimatorFactory,
    SklearnEstimator,
)
from sklearn.linear_model import LinearRegression


# =========================================================
# Fixtures
# =========================================================
@pytest.fixture
def sample_data():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    return X, y


# =========================================================
# MeanRegressor tests
# =========================================================
def test_mean_regressor_fit_predict(sample_data):
    X, y = sample_data

    model = MeanRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (len(X),)
    assert np.allclose(preds, np.mean(y))


def test_mean_regressor_not_fitted():
    model = MeanRegressor()

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((10, 1)))


# =========================================================
# Factory tests
# =========================================================
def test_factory_contains_mean():
    assert EstimatorFactory.contains("mean")
    assert EstimatorFactory.contains("mean_regressor")


def test_factory_create_mean():
    model = EstimatorFactory.create("mean")

    assert isinstance(model, MeanRegressor)


def test_factory_case_insensitive():
    m1 = EstimatorFactory.create("MeAn")
    m2 = EstimatorFactory.create("mean")

    assert type(m1) == type(m2)


# =========================================================
# Sklearn wrapper tests
# =========================================================
def test_sklearn_estimator_wrapper(sample_data):
    X, y = sample_data

    model = SklearnEstimator(LinearRegression)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_sklearn_predict_proba_not_available():
    model = SklearnEstimator(LinearRegression)

    with pytest.raises(NotImplementedError):
        model.predict_proba(np.zeros((5, 1)))


# =========================================================
# Integration-style test
# =========================================================
def test_estimator_interface_contract(sample_data):
    X, y = sample_data

    models = [
        MeanRegressor(),
        SklearnEstimator(LinearRegression),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(X)
        assert isinstance(preds, np.ndarray)
