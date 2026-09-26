import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kfc_procedure.core.steps.cstep import CStep


def test_cstep_mean_accepts_random_state(prediction_matrix, regression_targets):
    step = CStep(combiner="mean", task="regression", random_state=42)
    step.fit(prediction_matrix, regression_targets)
    assert type(step.strategy_).__name__ == "MeanCombiner"
    assert getattr(step.strategy_, "random_state", None) == 42
    np.testing.assert_allclose(step.predict(prediction_matrix), regression_targets)


def test_cstep_rejects_wrong_task_combiner(prediction_matrix, regression_targets):
    step = CStep(combiner="majority_vote", task="regression")
    with pytest.raises(ValueError):
        step.fit(prediction_matrix, regression_targets)


def test_cstep_predict_before_fit_raises(prediction_matrix):
    step = CStep(combiner="mean", task="regression")
    with pytest.raises(NotFittedError):
        step.predict(prediction_matrix)


def test_cstep_regression_predict_proba_raises(prediction_matrix, regression_targets):
    step = CStep(combiner="mean", task="regression")
    step.fit(prediction_matrix, regression_targets)
    with pytest.raises(AttributeError):
        step.predict_proba(prediction_matrix)
