import pytest
from sklearn.base import clone

from kfc_procedure.core.steps.cstep import CStep
from kfc_procedure.kfc import KFCProcedure, KFCRegressor, KFCClassifier


@pytest.mark.sklearn
def test_cstep_is_cloneable_without_mutating_none_params():
    step = CStep(combiner="mean", combiner_params=None, random_state=42)
    cloned = clone(step)
    assert cloned.combiner == "mean"
    assert cloned.combiner_params is None
    assert cloned.random_state == 42


@pytest.mark.sklearn
def test_kfc_procedure_is_cloneable():
    model = KFCProcedure(
        divergences=["euclidean"],
        local_model="linear_regression",
        combiner="mean",
        task="regression",
        divergences_params=None,
        local_model_params=None,
        combiner_params=None,
        random_state=42,
    )
    cloned = clone(model)
    assert cloned.random_state == 42


@pytest.mark.sklearn
@pytest.mark.parametrize("estimator_cls", [KFCRegressor, KFCClassifier])
def test_specialized_estimators_expose_explicit_sklearn_parameters(estimator_cls):
    # This intentionally catches *args/**kwargs constructors, which sklearn rejects.
    estimator = estimator_cls(
        divergences=["euclidean"],
        local_model="linear_regression" if estimator_cls is KFCRegressor else "logistic_regression",
        combiner="mean" if estimator_cls is KFCRegressor else "majority_vote",
    )
    params = estimator.get_params()
    assert "divergences" in params
    assert "local_model" in params
    assert "combiner" in params
