import numpy as np
import pytest
from sklearn.base import clone

from kfc_procedure.core.combiner.regression.mean import MeanCombiner
from kfc_procedure.core.combiner.regression.weighted_mean import WeightedMeanCombiner
from kfc_procedure.core.combiner.regression.stacking import StackingRegressorCombiner


def test_mean_combiner_predicts_row_mean():
    X = np.array([[1.0, 3.0], [2.0, 4.0]])
    model = MeanCombiner().fit(X)
    np.testing.assert_allclose(model.predict(X), [2.0, 3.0])


def test_mean_combiner_rejects_non_2d():
    with pytest.raises(ValueError):
        MeanCombiner().predict(np.array([1.0, 2.0]))


def test_mean_combiner_accepts_random_state_contract():
    model = MeanCombiner(random_state=42)
    assert model.random_state == 42
    cloned = clone(model)
    assert cloned.random_state == 42


def test_weighted_mean_learns_linear_relationship():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1]
    model = WeightedMeanCombiner(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


def test_stacking_regressor_basic_shape():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([1.0, 1.0, 3.0, 5.0])
    model = StackingRegressorCombiner().fit(X, y)
    assert model.predict(X).shape == (4,)
