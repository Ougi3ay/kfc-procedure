import numpy as np
from sklearn.linear_model import LinearRegression

from kfc_procedure.core.steps.fstep import FStep


def _extract_models(step, divergence="euclidean"):
    return [meta["model"] for meta in step.models_[divergence].values()]


def test_fstep_clones_estimator_instance_per_cluster():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0.0, 1.0, 10.0, 11.0])
    clusters = {"euclidean": np.array([0, 0, 1, 1])}

    original = LinearRegression()
    step = FStep(local_model=original, task="regression")
    step.fit(X, y, clusters)

    models = _extract_models(step)
    assert len(models) == 2
    assert models[0] is not models[1]
    assert all(model is not original for model in models)


def test_fstep_predict_shape_with_factory_model():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0.0, 1.0, 10.0, 11.0])
    clusters = {"euclidean": np.array([0, 0, 1, 1])}

    step = FStep(local_model="linear_regression", task="regression", random_state=42)
    step.fit(X, y, clusters)
    pred = step.predict(X, clusters)
    assert pred.shape == (4, 1)
    assert np.isfinite(pred).all()
